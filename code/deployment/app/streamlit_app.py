import os
import streamlit as st
import requests

API_URL = os.environ.get("API_URL", "http://0.0.0.0:8000")

def check_health():
    response = requests.get(f"{API_URL}/health")
    return response.json()

def predict(data):
    response = requests.post(f"{API_URL}/predict", json=data)
    return response.json()

st.title("Diabetes Predictor")

with st.expander("Health Check", expanded=False):
    if st.button("Check Health"):
        health_status = check_health()
        st.write("Health Status:", health_status)

with st.expander("Predict", expanded=False):
    with st.form("predict_form"):
        age = st.number_input("Age", value=0.0381, step=0.0001, format="%.4f")
        sex = st.number_input("Sex", value=0.0507, step=0.0001, format="%.4f")
        bmi = st.number_input("BMI", value=0.0617, step=0.0001, format="%.4f")
        bp = st.number_input("Blood Pressure", value=0.0219, step=0.0001, format="%.4f")
        s1 = st.number_input("S1", value=-0.0442, step=0.0001, format="%.4f")
        s2 = st.number_input("S2", value=-0.0348, step=0.0001, format="%.4f")
        s3 = st.number_input("S3", value=-0.0434, step=0.0001, format="%.4f")
        s4 = st.number_input("S4", value=-0.0026, step=0.0001, format="%.4f")
        s5 = st.number_input("S5", value=0.0199, step=0.0001, format="%.4f")
        s6 = st.number_input("S6", value=-0.0176, step=0.0001, format="%.4f")

        submitted = st.form_submit_button("Submit")
        if submitted:
            features = [age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]
            data = {"data": [features]}
            result = predict(data)
            if result and "predictions" in result:
                st.success(f"Prediction: {result['predictions'][0]:.4f}")