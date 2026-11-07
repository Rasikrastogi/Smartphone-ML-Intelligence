"""
Performance Predictor page: estimates the AnTuTu score from processor and memory.
"""

import streamlit as st

import prediction_models as pm
from form_helpers import assets_or_stop, choose, number, processor_choice, show_filled

models, metadata = assets_or_stop()

st.title("Performance Predictor")
st.write(
    "Estimate a phone's AnTuTu benchmark score from its processor and memory. "
    "Every field can be left empty; empty fields use a typical value."
)

inputs = {}

with st.expander("Processor", expanded=True):
    inputs["processor"] = processor_choice(metadata, "performance_processor")
    processor = pm.processor_details(inputs["processor"], metadata) if inputs["processor"] else None

    if processor:
        st.caption(
            f"Filled in automatically: {processor['processor_type']}, {processor['fab']} nm. "
            "Skipped fields below use what is typical for this processor."
        )
        typical = processor["typical"]
    else:
        st.caption(
            "Processor not listed or not known? Leave it empty and choose the processor type and "
            "fabrication size instead. The estimate is then less precise."
        )
        left, right = st.columns(2)
        with left:
            inputs["processor_type"] = choose("processor_type", metadata, "performance_processor_type")
        with right:
            inputs["fab"] = choose("fab", metadata, "performance_fab")
        typical = {}

with st.expander("Memory and body", expanded=True):
    left, right = st.columns(2)
    with left:
        inputs["ram_type"] = choose("ram_type", metadata, "performance_ram", typical=typical.get("ram_type"))
        inputs["size"] = number("size", metadata, "performance_size", step=0.01, decimals=True,
                                typical=typical.get("size"))
    with right:
        inputs["rom_type"] = choose("rom_type", metadata, "performance_rom", typical=typical.get("rom_type"))
        inputs["thickness"] = number("thickness", metadata, "performance_thickness", step=0.1, decimals=True,
                                     typical=typical.get("thickness"))

with st.expander("Geekbench scores (optional)", expanded=True):
    st.caption(
        "If you know both Geekbench scores, a more accurate model is used. "
        "If either is left empty, the hardware-only model is used."
    )
    left, right = st.columns(2)
    with left:
        inputs["geekbench_single"] = st.number_input(
            "Geekbench single-core", min_value=0, value=None, step=50,
            placeholder="Optional", key="performance_gb_single"
        )
    with right:
        inputs["geekbench_multiple"] = st.number_input(
            "Geekbench multi-core", min_value=0, value=None, step=100,
            placeholder="Optional", key="performance_gb_multi"
        )

if st.button("Predict performance", type="primary"):

    result = pm.predict_performance(inputs, models, metadata)

    score = result["prediction"]
    error = result["typical_error"]

    st.metric("Estimated AnTuTu score", f"{score:,.0f}")
    st.caption(
        f"Likely between {max(0, score - error):,.0f} and {score + error:,.0f} "
        f"(typical error: ± {error:,.0f}). Model: {result['model']}."
    )

    if not processor:
        st.info(
            "Without a known processor, the model relies on processor type, fabrication size, and memory. "
            "The typical error shown is the one measured for processors the model has never seen."
        )

    show_filled(result["filled"], metadata)
