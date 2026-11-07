"""
Shared input widgets for the predictor pages.

Every widget starts empty. An empty widget means "skip": the page shows the
typical value that will be used instead, and the prediction code fills it in.
"""

import streamlit as st

import prediction_models as pm


@st.cache_resource
def load_assets():
    return pm.load_assets()


def assets_or_stop():
    try:
        return load_assets()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()


# --------------------------------------------------------------------------- display formats

def yes_no(value):
    return "Yes" if value else "No"


def video_label(value):
    """'4, 60' -> '4K at 60 fps', '1, 30' -> '1080p at 30 fps'."""
    resolution, fps = [part.strip() for part in str(value).split(",")]
    names = {"0.44": "480p", "1": "1080p", "2": "2K", "4": "4K", "8": "8K"}
    return f"{names.get(resolution, resolution + 'K')} at {fps} fps"


def protection_label(value):
    """'Corning, GG 7i' -> 'Corning GG 7i', 'Panda Glass, ' -> 'Panda Glass'."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return " ".join(parts)


def sale_label(value):
    return {
        "Online": "Online only",
        "Offline": "Offline (retail stores)",
        "Both": "Online and offline",
        "Flagship": "Flagship",
    }.get(value, value)


def megapixels(value):
    return f"{value:g} MP"


FORMATS = {
    "curved_display": yes_no,
    "wireless_charging": yes_no,
    "rear_video": video_label,
    "front_video": video_label,
    "protection": protection_label,
    "sale": sale_label,
    "ip_rating": lambda value: f"IP{value}",
    "fab": lambda value: f"{value} nm",
    "refresh_rate": lambda value: f"{value} Hz",
    "main": megapixels,
    "front": megapixels,
    "ultrawide": megapixels,
}


# --------------------------------------------------------------------------- widgets

def choose(name, metadata, key, options=None, label=None, typical=None, fmt=None):
    """Dropdown that starts empty; empty means skip."""

    options = options if options is not None else metadata["options"][name]
    label = label or metadata["input_labels"].get(name, name)
    fmt = fmt or FORMATS.get(name, str)
    typical = metadata["defaults"].get(name) if typical is None else typical

    placeholder = f"Skip — typical: {fmt(typical)}" if typical is not None else "Skip"

    return st.selectbox(label, options, index=None, placeholder=placeholder, format_func=fmt, key=key)


def yes_no_choice(name, metadata, key, label=None, typical=None):
    return choose(name, metadata, key, options=[True, False], label=label, typical=typical, fmt=yes_no)


def number(name, metadata, key, step, label=None, typical=None, decimals=False):
    """Number box that starts empty; empty means skip."""

    label = label or metadata["input_labels"].get(name, name)
    typical = metadata["defaults"].get(name) if typical is None else typical

    if decimals:
        return st.number_input(
            label, min_value=0.0, value=None, step=float(step), format="%.2f",
            placeholder=f"Skip — typical: {typical:g}", key=key
        )

    return st.number_input(
        label, min_value=0, value=None, step=int(step),
        placeholder=f"Skip — typical: {typical:,.0f}", key=key
    )


def processor_choice(metadata, key):
    labels = [entry["label"] for entry in metadata["processors"]]
    return st.selectbox(
        "Processor",
        labels,
        index=None,
        placeholder="Skip, or type to search (e.g. Snapdragon 8 Elite)",
        key=key
    )


def show_filled(filled, metadata):
    """List the typical values used for skipped inputs, in the same format as the dropdowns."""

    if not filled:
        st.caption("All inputs were filled in; no typical values were needed.")
        return

    name_for_label = {label: name for name, label in metadata["input_labels"].items()}

    with st.expander(f"Typical values used for {len(filled)} skipped inputs"):
        for label, value in filled.items():
            fmt = FORMATS.get(name_for_label.get(label))
            if fmt and not isinstance(value, str):
                value = fmt(value)
            elif fmt and name_for_label.get(label) in ("rear_video", "front_video", "protection", "sale"):
                value = fmt(value)
            st.markdown(f"- **{label}:** {value}")
