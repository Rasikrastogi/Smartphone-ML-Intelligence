"""
Price Predictor page: estimates a phone's price from its specifications.
"""

import streamlit as st

import prediction_models as pm
from form_helpers import (
    assets_or_stop, choose, megapixels, number, processor_choice, show_filled, yes_no, yes_no_choice
)

models, metadata = assets_or_stop()
defaults = metadata["defaults"]
camera = metadata["camera"]

st.title("Price Predictor")
st.write(
    "Estimate a phone's price from its specifications. Fill in only what you know: "
    "every field can be left empty, and empty fields use a typical value."
)

inputs = {}

# --------------------------------------------------------------------------- basics
with st.expander("Brand and launch", expanded=True):
    left, right = st.columns(2)
    with left:
        inputs["brand"] = choose("brand", metadata, "price_brand")
        inputs["launch_date"] = st.date_input("Launch date", value=None, format="DD.MM.YYYY",
                                              help="Leave empty for a phone launching now.", key="price_launch")
    with right:
        inputs["sale"] = choose("sale", metadata, "price_sale")

# --------------------------------------------------------------------------- processor and memory
with st.expander("Processor and memory", expanded=True):
    inputs["processor"] = processor_choice(metadata, "price_processor")
    processor = pm.processor_details(inputs["processor"], metadata) if inputs["processor"] else None

    if processor:
        st.caption(
            f"Filled in automatically: {processor['processor_type']}, {processor['fab']} nm. "
            "Skipped memory types use what is typical for this processor."
        )
        typical_memory = processor["typical"]
    else:
        left, right = st.columns(2)
        with left:
            inputs["processor_type"] = choose("processor_type", metadata, "price_processor_type")
        with right:
            inputs["fab"] = choose("fab", metadata, "price_fab")
        typical_memory = {}

    left, right = st.columns(2)
    with left:
        inputs["ram_type"] = choose("ram_type", metadata, "price_ram", typical=typical_memory.get("ram_type"))
    with right:
        inputs["rom_type"] = choose("rom_type", metadata, "price_rom", typical=typical_memory.get("rom_type"))

# --------------------------------------------------------------------------- display
with st.expander("Display"):
    left, right = st.columns(2)
    with left:
        inputs["display_type"] = choose("display_type", metadata, "price_display_type")
        inputs["size"] = number("size", metadata, "price_size", step=0.01, decimals=True,
                                typical=typical_memory.get("size"))
        inputs["hbm"] = number("hbm", metadata, "price_hbm", step=50)
        inputs["curved_display"] = yes_no_choice("curved_display", metadata, "price_curved")
    with right:
        inputs["refresh_rate"] = choose("refresh_rate", metadata, "price_refresh")
        inputs["ppi"] = number("ppi", metadata, "price_ppi", step=10)
        inputs["protection"] = choose("protection", metadata, "price_protection")

# --------------------------------------------------------------------------- battery
with st.expander("Battery and charging"):
    left, right = st.columns(2)
    with left:
        inputs["battery"] = number("battery", metadata, "price_battery", step=100)
        inputs["wireless_charging"] = yes_no_choice("wireless_charging", metadata, "price_wireless")
    with right:
        inputs["charging_speed"] = number("charging_speed", metadata, "price_charging", step=5)

# --------------------------------------------------------------------------- camera
with st.expander("Camera", expanded=True):
    left, right = st.columns(2)
    with left:
        inputs["useful_camera_number"] = choose(
            "useful_camera_number", metadata, "price_useful_cameras",
            label="Useful rear cameras (not counting depth/macro lenses)"
        )
    with right:
        inputs["main"] = choose("main", metadata, "price_main")

    useful = inputs["useful_camera_number"] or defaults["useful_camera_number"]

    if inputs["useful_camera_number"] is None:
        st.caption(f"Using {useful} useful cameras (typical) for the questions below.")

    zoom_format = lambda value: f"{value:g}x"

    if useful == 2:
        left, right = st.columns(2)
        with left:
            inputs["has_ultrawide"] = yes_no_choice(
                "has_ultrawide", metadata, "price_has_ultrawide",
                label="Is the second camera an ultrawide?", typical=True
            )
        with right:
            if inputs["has_ultrawide"] is False:
                inputs["zoom"] = choose("zoom", metadata, "price_zoom", options=camera["zoom_options"],
                                        label="Telephoto zoom", typical=camera["zoom_default"], fmt=zoom_format)
            else:
                inputs["ultrawide"] = choose("ultrawide", metadata, "price_ultrawide",
                                             label="Ultrawide camera (MP)",
                                             typical=camera["ultrawide_mp_default"], fmt=megapixels)

    elif useful >= 3:
        left, right = st.columns(2)
        with left:
            inputs["ultrawide"] = choose("ultrawide", metadata, "price_ultrawide",
                                         label="Ultrawide camera (MP)",
                                         typical=camera["ultrawide_mp_default"], fmt=megapixels)
        with right:
            inputs["zoom"] = choose("zoom", metadata, "price_zoom", options=camera["zoom_options"],
                                    label="Telephoto zoom (e.g. 3.5x)", typical=camera["zoom_default"],
                                    fmt=zoom_format)

    left, right = st.columns(2)
    with left:
        inputs["front"] = choose("front", metadata, "price_front")
        inputs["rear_video"] = choose("rear_video", metadata, "price_rear_video")
    with right:
        inputs["front_video"] = choose("front_video", metadata, "price_front_video")

# --------------------------------------------------------------------------- build
with st.expander("Build and sound"):
    left, right = st.columns(2)
    with left:
        inputs["thickness"] = number("thickness", metadata, "price_thickness", step=0.1, decimals=True,
                                     typical=typical_memory.get("thickness"))
        inputs["back_material"] = choose("back_material", metadata, "price_back")
        inputs["ip_rating"] = choose("ip_rating", metadata, "price_ip")
    with right:
        inputs["weight"] = number("weight", metadata, "price_weight", step=5)
        inputs["frame_material"] = choose("frame_material", metadata, "price_frame")
        inputs["speaker"] = choose("speaker", metadata, "price_speaker")

# --------------------------------------------------------------------------- prediction
if st.button("Predict price", type="primary"):

    result = pm.predict_price(inputs, models, metadata)

    price = result["prediction"]
    error = result["typical_error"]

    st.metric("Estimated price", f"Rs. {price:,.0f}")
    st.caption(
        f"Likely between Rs. {max(0, price - error):,.0f} and Rs. {price + error:,.0f} "
        f"(typical error of the model: ± Rs. {error:,.0f}, from cross-validation)."
    )

    show_filled(result["filled"], metadata)
