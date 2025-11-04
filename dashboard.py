
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

# Load model and encoders
model = joblib.load("model_artifacts/crop_yield_lgbm.pkl")
encoders = joblib.load("model_artifacts/encoders.joblib")

st.title("🌾 Crop Yield Prediction Dashboard")

# Inputs
state = st.selectbox("Select State", encoders['State'].classes_)
crop = st.selectbox("Select Crop", encoders['Crop'].classes_)
season = st.selectbox("Select Season", encoders['Season'].classes_)
area = st.number_input("Area (hectares)", 0.1, 100000.0, 500.0)
rainfall = st.number_input("Annual Rainfall (mm)", 0.0, 10000.0, 800.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", 0.0, 10000.0, 200.0)
pesticide = st.number_input("Pesticide (kg/ha)", 0.0, 10000.0, 50.0)
crop_year = st.slider("Crop Year", 1990, 2025, 2020)

# Derived features
rainfall_per_area = rainfall / (area + 1e-6)
fertilizer_per_area = fertilizer / (area + 1e-6)
pesticide_per_area = pesticide / (area + 1e-6)

# Encode categorical
state_enc = encoders['State'].transform([state])[0]
crop_enc = encoders['Crop'].transform([crop])[0]
season_enc = encoders['Season'].transform([season])[0]

# DataFrame for prediction
X_input = pd.DataFrame([{
    "Crop": crop_enc,
    "Crop_Year": crop_year,
    "Season": season_enc,
    "State": state_enc,
    "Area": area,
    "Production": fertilizer + pesticide + rainfall,  # simple proxy
    "Annual_Rainfall": rainfall,
    "Fertilizer": fertilizer,
    "Pesticide": pesticide,
    "Rainfall_per_Area": rainfall_per_area,
    "Fertilizer_per_Area": fertilizer_per_area,
    "Pesticide_per_Area": pesticide_per_area
}])

# Predict
if st.button("Predict Yield"):
    pred = model.predict(X_input)[0]
    st.success(f"🌾 Predicted Crop Yield: {pred:.2f} tons/ha")

    st.write("### Feature Impact")
    st.bar_chart(X_input.T)

st.caption("Model: LightGBM Regressor trained on Indian Crop Yield dataset")
