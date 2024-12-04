import streamlit as st
import pandas as pd
import joblib
import gdown
from sklearn.preprocessing import StandardScaler

# URLs for the scaler and the logistic regression model
scaler_url = "https://drive.google.com/uc?id=1KIj_PCCiAmOMDi6nZj-XqpFzAAsC_a5H"
model_url = "https://drive.google.com/uc?id=1n-oQezif9IMaTpdmw5cPas1aFOWOlygy"

# Local file paths
scaler_path = "scaler.pkl"
model_path = "model.pkl"

# Download files from Google Drive
st.write("Downloading required files...")
gdown.download(scaler_url, scaler_path, quiet=False)
gdown.download(model_url, model_path, quiet=False)

# Load the scaler and model
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

# App title and description
st.title("NALYTICS Stroke Prediction by Maxwell Adigwe")
st.sidebar.markdown(""" 
This app predicts the likelihood of stroke based on user input data. 
                     
Please use the options below to provide your details.
""")

# Function to collect user input features
def user_input_features():
    gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
    age = st.sidebar.slider("Age", 0, 100, 50)
    hypertension = st.sidebar.selectbox("Hypertension", (0, 1))
    heart_disease = st.sidebar.selectbox("Heart Disease", (0, 1))
    ever_married = st.sidebar.selectbox("Ever Married", ("No", "Yes"))
    work_type = st.sidebar.selectbox(
        "Work Type",
        ("Private", "Self-employed", "Govt_job", "Children", "Never_worked")
    )
    residence_type = st.sidebar.selectbox("Residence Type", ("Urban", "Rural"))
    avg_glucose_level = st.sidebar.slider("Average Glucose Level", 50.0, 300.0, 120.0)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    smoking_status = st.sidebar.selectbox(
        "Smoking Status", ("never smoked", "formerly smoked", "smokes", "Unknown")
    )

    # Convert user inputs to a DataFrame
    data = {
        "gender": 0 if gender == "Male" else 1,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": 1 if ever_married == "Yes" else 0,
        "work_type": {
            "Private": 0,
            "Self-employed": 1,
            "Govt_job": 2,
            "Children": 3,
            "Never_worked": 4,
        }[work_type],
        "Residence_type": 0 if residence_type == "Urban" else 1,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "smoking_status": {
            "never smoked": 0,
            "formerly smoked": 1,
            "smokes": 2,
            "Unknown": 3,
        }[smoking_status],
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Collect user input
input_df = user_input_features()

# Display user inputs
st.subheader("User Input Parameters")
st.write(input_df)

# Scale the input data
scaled_input = scaler.transform(input_df)

# Make predictions




# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Note:** This app uses a trained logistic regression model to predict stroke likelihood based on historical data.")

