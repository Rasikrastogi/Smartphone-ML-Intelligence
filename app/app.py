"""
Smartphone ML Intelligence app.

Run from the project root:
    streamlit run app/app.py

Pages:
    Smartphone Finder      - recommendations from a sentence (notebook 08)
    Price Predictor        - price estimate from specifications (notebook 09)
    Performance Predictor  - AnTuTu estimate from processor and memory (notebook 09)
"""

import streamlit as st

st.set_page_config(page_title="Smartphone ML Intelligence", layout="centered")

navigation = st.navigation([
    st.Page("finder.py", title="Smartphone Finder", default=True),
    st.Page("price_predictor.py", title="Price Predictor"),
    st.Page("performance_predictor.py", title="Performance Predictor"),
])

navigation.run()
