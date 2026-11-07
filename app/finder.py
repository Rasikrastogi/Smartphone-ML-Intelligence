"""
Smartphone Finder page: recommendations from a sentence (notebook 08 pipeline).
"""

import streamlit as st

import smartphone_assistant as assistant

COLUMN_LABELS = {
    "name": "Phone",
    "brand": "Brand",
    "price": "Price (Rs.)",
    "value_score": "Value score",
    "battery": "Battery (mAh)",
    "antutu": "AnTuTu",
    "useful_camera_number": "Useful cameras",
    "charging_speed": "Charging (W)",
    "hbm": "Peak brightness (nits)",
    "hbm_filled": "Peak brightness (nits, filled)",
    "refresh_rate": "Refresh rate (Hz)",
    "thickness": "Thickness (mm)",
    "weight": "Weight (g)",
    "size": "Screen (inches)",
    "ppi": "Pixel density (ppi)",
    "telephoto_zoom": "Telephoto zoom (x)",
    "brand_value": "Brand value (Rs.)",
    "note": "Note",
}

EXAMPLES = [
    "waterproof phone with a good camera under 40k",
    "I binge watch movies, budget is 35k, not a Samsung",
    "need a phone for PUBG between 25k and 35k",
    "I need an iPhone under 40000",
]


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")

    use_model = st.toggle(
        "Use Ollama for paraphrases",
        value=True,
        help="The model is only asked when a request contains no feature word, "
             "for example 'a phone for PUBG'."
    )
    model = st.text_input("Ollama model", value=assistant.DEFAULT_MODEL, disabled=not use_model)

    available = int((assistant.data["availability"] == 1).sum())
    st.caption(f"{available} available phones in the dataset.")
    st.caption(
        "Budgets: 'under X' searches 0.8X to X, plus phones up to 1.1X with a discount note; "
        "'around X' searches ±10%; 'between X and Y' searches X to Y."
    )

# --------------------------------------------------------------------------- main
st.title("Smartphone Finder")
st.write("Describe the phone you want in your own words.")

with st.form("request"):
    sentence = st.text_input(
        "Your request",
        placeholder="e.g. waterproof phone with a good camera under 40k"
    )
    submitted = st.form_submit_button("Find phones", type="primary")

st.caption("Examples: " + " · ".join(f"*{example}*" for example in EXAMPLES))

if submitted and sentence.strip():

    try:
        request, note = assistant.extract_request(sentence, model=model if use_model else None)
    except Exception as error:
        st.warning(
            f"Could not reach the Ollama model ({error}). "
            "Showing results from keywords only. Start Ollama or switch the model off in the sidebar."
        )
        request, note = assistant.extract_request(sentence, model=None)

    st.subheader("Understood as")
    for label, value in assistant.describe_request(request, note):
        st.markdown(f"**{label}:** {value}")

    st.subheader("Recommendations")
    result = assistant.recommend(request)

    if result.no_match:
        first_line, *reasons = result.no_match.split("\n")
        st.error(first_line)
        for reason in reasons:
            st.markdown(reason)
    else:
        st.caption(result.summary)
        table = result.table.rename(columns=COLUMN_LABELS)
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Price (Rs.)": st.column_config.NumberColumn(format="%d"),
                "Value score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            }
        )

elif submitted:
    st.info("Type a request first.")
