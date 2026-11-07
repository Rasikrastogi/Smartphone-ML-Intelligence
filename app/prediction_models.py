"""
Price and performance prediction: turns simple user inputs into the feature rows
the models from notebook 09 expect, and returns predictions.

The models and the app metadata (typical values, dropdown options, processor
details, camera patterns) are created by notebooks/09_final_models.ipynb and
stored in the models/ folder.
"""

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

MODEL_FILES = {
    "price": "price_model.joblib",
    "performance_hardware": "performance_hardware_model.joblib",
    "performance_geekbench": "performance_geekbench_model.joblib",
}
METADATA_FILE = "app_metadata.json"


def load_assets(models_dir=MODELS_DIR):
    """Load the three models and the metadata. Raises FileNotFoundError if notebook 09 has not been run."""

    models_dir = Path(models_dir)
    missing = [name for name in list(MODEL_FILES.values()) + [METADATA_FILE] if not (models_dir / name).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing in {models_dir}: {', '.join(missing)}. Run notebooks/09_final_models.ipynb first."
        )

    models = {key: joblib.load(models_dir / filename) for key, filename in MODEL_FILES.items()}
    metadata = json.loads((models_dir / METADATA_FILE).read_text(encoding="utf-8"))

    return models, metadata


# =========================================================================== camera rule

def telephoto_label(zoom):
    """3.5 -> 'Telephoto - 3.5x', 3.0 -> 'Telephoto - 3x' (the format used in the dataset)."""
    return f"Telephoto - {float(zoom):g}x"


def camera_fields(useful_cameras, main_mp, has_ultrawide, ultrawide_mp, zoom, metadata):
    """Build every camera column of the price model from the few camera inputs.

    useful_cameras 1: main camera only
    useful_cameras 2: main + ultrawide (or, without an ultrawide, main + telephoto)
    useful_cameras 3: main + ultrawide + telephoto (zoom)
    useful_cameras 4: as 3, plus the typical second telephoto of 4-camera phones

    Anything not asked (total camera count, extra low-resolution lenses, telephoto
    megapixels) comes from the most common pattern for that camera count.
    Returns (fields, filled), where filled lists the values taken from typical data.
    """

    camera = metadata["camera"]
    pattern = dict(camera["patterns"][str(useful_cameras)])
    filled = {}

    def typical(name, value, default):
        if value is None:
            filled[name] = default
            return default
        return value

    fields = {
        "useful_camera_number": useful_cameras,
        "main": typical("Main camera (MP)", main_mp, metadata["defaults"]["main"]),
        "camera_number": pattern["camera_number"],
        "ultrawide": np.nan,
        "third": pattern["third"],
        "third_mp": pattern["third_mp"],
        "fourth": pattern["fourth"],
        "fourth_mp": pattern["fourth_mp"],
    }

    wants_ultrawide = useful_cameras >= 3 or (
        useful_cameras == 2 and typical("Has ultrawide", has_ultrawide, True)
    )

    if wants_ultrawide:
        fields["ultrawide"] = typical("Ultrawide camera (MP)", ultrawide_mp, camera["ultrawide_mp_default"])

    needs_telephoto = useful_cameras >= 3 or (useful_cameras == 2 and not wants_ultrawide)

    if needs_telephoto:
        zoom_value = typical("Telephoto zoom (x)", zoom, camera["zoom_default"])
        fields["third"] = telephoto_label(zoom_value)
        fields["third_mp"] = camera["telephoto_mp_default"]
        if useful_cameras == 2:
            fields["camera_number"] = max(2, pattern["camera_number"])

    fields["has_ultrawide"] = bool(wants_ultrawide)
    fields["has_third_camera"] = fields["third"] is not None and not pd.isna(fields["third"])
    fields["has_fourth_camera"] = fields["fourth"] is not None and not pd.isna(fields["fourth"])

    return fields, filled


# =========================================================================== feature rows

def _to_frame(row, features):
    """One-row DataFrame in the model's column order; missing categories as NaN (as in training)."""

    frame = pd.DataFrame([{feature: row.get(feature, np.nan) for feature in features}])
    return frame.astype(object).where(frame.notna(), np.nan)


def _fill(inputs, name, label, metadata, filled, processor=None):
    """The user's value, or the typical value (median / mode) if the input was skipped.

    If a processor was chosen and it has a typical value for this input (RAM type,
    storage type, size, thickness), the typical value among phones with that
    processor is used instead of the overall one.
    """

    value = inputs.get(name)

    if value is None:
        if processor and name in processor.get("typical", {}):
            value = processor["typical"][name]
            filled[label] = f"{value} (typical for {processor['label']})"
        else:
            value = metadata["defaults"][name]
            filled[label] = value

    return value


def processor_details(processor_label, metadata):
    """'Snapdragon 6 Gen 3' -> {'processor': '6 Gen 3', 'processor_type': 'Snapdragon', 'fab': 4}."""
    for entry in metadata["processors"]:
        if entry["label"] == processor_label:
            return entry
    return None


def build_price_row(inputs, metadata, today=None):
    """Turn the price page inputs into the price model's feature row.

    `inputs` maps input names to values; None means "skipped".
    Returns (row, filled) where filled shows which typical values were used.
    """

    today = today or date.today()
    filled = {}
    row = {}

    # Processor: only its type and fabrication size are price features
    processor = processor_details(inputs.get("processor"), metadata) if inputs.get("processor") else None
    row["processor_type"] = processor["processor_type"] if processor else _fill(
        inputs, "processor_type", "Processor type", metadata, filled)
    row["fab"] = processor["fab"] if processor else _fill(inputs, "fab", "Fabrication (nm)", metadata, filled)

    labels = metadata["input_labels"]
    for name in metadata["price"]["simple_inputs"]:
        row[name] = _fill(inputs, name, labels.get(name, name), metadata, filled, processor)

    # Camera
    fields, camera_filled = camera_fields(
        useful_cameras=inputs.get("useful_camera_number") or metadata["defaults"]["useful_camera_number"],
        main_mp=inputs.get("main"),
        has_ultrawide=inputs.get("has_ultrawide"),
        ultrawide_mp=inputs.get("ultrawide"),
        zoom=inputs.get("zoom"),
        metadata=metadata,
    )
    if inputs.get("useful_camera_number") is None:
        filled["Useful cameras"] = metadata["defaults"]["useful_camera_number"]
    row.update(fields)
    filled.update(camera_filled)

    # Launch date -> launch year and age; a phone is assumed to be on sale now
    launch = inputs.get("launch_date")
    if launch is None:
        launch = today
        filled["Launch date"] = f"today ({today.isoformat()})"
    row["launch_year"] = launch.year
    row["phone_age_days"] = max(0, (today - launch).days)
    row["availability"] = 1

    for name in ["curved_display", "wireless_charging", "has_ultrawide", "has_third_camera", "has_fourth_camera"]:
        row[name] = bool(row[name])

    return _to_frame(row, metadata["price"]["features"]), filled


def build_performance_row(inputs, metadata):
    """Turn the performance page inputs into a feature row.

    A skipped processor is treated as unknown (the model then relies on processor
    type, fabrication, and memory), rather than replaced by the most common chip.
    Returns (row, filled, model_key, processor_known).
    """

    filled = {}
    row = {}

    processor = processor_details(inputs.get("processor"), metadata) if inputs.get("processor") else None

    if processor:
        row["processor"] = processor["processor"]
        row["processor_type"] = processor["processor_type"]
        row["fab"] = processor["fab"]
    else:
        row["processor"] = "Unknown"          # not in training data -> ignored by the encoder
        filled["Processor"] = "unknown (not used)"
        row["processor_type"] = _fill(inputs, "processor_type", "Processor type", metadata, filled)
        row["fab"] = _fill(inputs, "fab", "Fabrication (nm)", metadata, filled)

    labels = metadata["input_labels"]
    for name in ["ram_type", "rom_type", "size", "thickness"]:
        row[name] = _fill(inputs, name, labels.get(name, name), metadata, filled, processor)

    has_geekbench = inputs.get("geekbench_single") is not None and inputs.get("geekbench_multiple") is not None

    if has_geekbench:
        row["geekbench_single"] = inputs["geekbench_single"]
        row["geekbench_multiple"] = inputs["geekbench_multiple"]
        model_key = "performance_geekbench"
    else:
        model_key = "performance_hardware"

    features = metadata["performance"][model_key.split("_")[1]]["features"]

    return _to_frame(row, features), filled, model_key, processor is not None


# =========================================================================== predictions

def predict_price(inputs, models, metadata, today=None):

    row, filled = build_price_row(inputs, metadata, today)
    prediction = float(models["price"].predict(row)[0])
    typical_error = metadata["price"]["cv"]["mae"]

    return {"prediction": prediction, "typical_error": typical_error, "filled": filled, "row": row}


def predict_performance(inputs, models, metadata):

    row, filled, model_key, processor_known = build_performance_row(inputs, metadata)
    prediction = float(models[model_key].predict(row)[0])

    model_info = metadata["performance"][model_key.split("_")[1]]
    # Unknown processor: the processor-grouped error is the realistic one
    typical_error = model_info["cv"]["mae"] if processor_known else model_info["grouped_cv"]["mae"]

    return {
        "prediction": prediction,
        "typical_error": typical_error,
        "filled": filled,
        "model": model_info["name"],
        "row": row,
    }
