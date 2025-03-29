import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('../models/best_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

st.title('📊 Customer Churn Prediction')

# Define user input form
st.sidebar.header('Input Customer Data')

def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    MonthlyCharges = st.sidebar.number_input('Monthly Charges ($)', 10, 150, 50)
    TotalCharges = tenure * MonthlyCharges
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['Yes', 'No'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])

    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'SeniorCitizen_Yes': 1 if SeniorCitizen == 'Yes' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0,
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# Prediction
prediction_proba = model.predict_proba(input_df)[:,1][0]
prediction = model.predict(input_df)[0]

# Output
st.subheader('Prediction Result')
churn_label = 'Churn' if prediction == 1 else 'No Churn'
st.write(f"Prediction: **{churn_label}**")
st.write(f"Churn Probability: **{prediction_proba:.2%}**")
