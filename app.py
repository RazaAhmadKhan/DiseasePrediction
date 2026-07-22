"""
CodeAlpha_DiseasePrediction — Real-Time Prediction GUI
========================================================
Streamlit web app that loads the trained model (model.pkl + scaler.pkl)
produced by train_model.py and lets a user enter a patient's data to get
a live diabetes risk prediction.

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load trained model, scaler, and metadata
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not (os.path.exists("model.pkl") and os.path.exists("scaler.pkl")):
        return None, None, None
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata

model, scaler, metadata = load_artifacts()

if model is None:
    st.error("Model files not found. Please run `python train_model.py` first "
             "to download the dataset and train the model.")
    st.stop()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🩺 Diabetes Risk Prediction")
st.caption("CodeAlpha ML Internship — Task 4: Disease Prediction from Medical Data")

with st.expander("ℹ️ About this model"):
    st.write(
        f"**Best model selected:** {metadata['best_model_name']}\n\n"
        f"**Trained on:** Pima Indians Diabetes Dataset (UCI ML Repository)\n\n"
        f"**Test-set performance:**"
    )
    m = metadata["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{m['Accuracy']*100:.1f}%")
    c2.metric("Precision", f"{m['Precision']*100:.1f}%")
    c3.metric("Recall", f"{m['Recall']*100:.1f}%")
    c4.metric("F1-Score", f"{m['F1-Score']*100:.1f}%")
    c5.metric("ROC-AUC", f"{m['ROC-AUC']:.3f}")

st.divider()
st.subheader("Enter Patient Data")
st.write("Fill in the fields below and click **Predict** to get a real-time risk assessment.")

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)

    with col2:
        insulin = st.number_input("Insulin level (mu U/mL)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    # Treat 0 entries for medically-impossible fields the same way training did
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        if input_dict[col] == 0:
            input_dict[col] = metadata["feature_medians"][col]

    input_df = pd.DataFrame([input_dict])[metadata["feature_names"]]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Risk of Diabetes** — estimated probability: {probability*100:.1f}%")
    else:
        st.success(f"✅ **Low Risk of Diabetes** — estimated probability: {probability*100:.1f}%")

    st.progress(min(float(probability), 1.0))
    st.caption(
        "This is a machine learning estimate based on the Pima Indians Diabetes "
        "dataset and is **not a medical diagnosis**. Please consult a healthcare "
        "professional for an accurate assessment."
    )

    with st.expander("View input data sent to the model"):
        st.dataframe(input_df, use_container_width=True)

st.divider()
st.caption("Built by Raza Ahmad Khan · CodeAlpha Machine Learning Internship · Task 4")
