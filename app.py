import streamlit as st
import joblib
import numpy as np

# Load scaler and model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("🔍 Customer Churn Prediction App")

st.divider()
st.write("Please enter the values and click **Predict** to get the churn prediction.")
st.divider()

# Inputs
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)

gender = st.selectbox("Select Gender", ["Male", "Female"])

tenure = st.number_input("Enter Tenure (months)", min_value=0, max_value=130, value=10)

monthlycharge = st.number_input(
    "Enter Monthly Charge",
    min_value=30.0,
    max_value=150.0,
    value=50.0
)

st.divider()

predictbutton = st.button("Predict")

if predictbutton:
    # Encode gender
    gender_selected = 1 if gender == "Female" else 0

    # Order: age, gender, tenure, monthlycharges
    x = np.array([[age, gender_selected, tenure, monthlycharge]])

    # Scale input
    x_scaled = scaler.transform(x)

    # Predict
    prediction = model.predict(x_scaled)

    # Output
    result = "Yes (Customer will Churn)" if prediction == 1 else "No (Customer will NOT Churn)"
    st.success(f"Prediction: **{result}**")

else:
    st.info("👆 Please enter the values and click **Predict**")
