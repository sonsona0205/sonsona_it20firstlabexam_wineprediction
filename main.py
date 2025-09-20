import streamlit as st
import pandas as pd
import joblib

model = joblib.load("wine_quality_best_model.joblib")
features = joblib.load("wine_feature_names.joblib")

st.title("Wine Quality - Good or Not (probability)")

inputs = {}
for feat in features:
    val = st.number_input(feat, value=float(0.0))
    inputs[feat] = val

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=features)
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0, 1]
    st.write("**Prediction:**", "Good (>=7)" if pred==1 else "Not good (<7)")
    st.write(f"**Confidence (probability of good):** {proba:.3f}")