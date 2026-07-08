import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.title("🌱 Crop Recommendation System")

st.write("Enter soil and climate values")

# Inputs
N = st.number_input("Nitrogen (N)", 0, 140)
P = st.number_input("Phosphorus (P)", 0, 145)
K = st.number_input("Potassium (K)", 0, 205)

temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

# Prediction
if st.button("Recommend Crop"):

    features = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall]
    })

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)

    st.success(f"Recommended Crop: {prediction[0]}")