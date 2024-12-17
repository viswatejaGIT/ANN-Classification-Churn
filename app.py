import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the saved files
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Load the trained ANN model
model = load_model('model.h5')

# Streamlit UI
st.title("Customer Churn Prediction App")

# Input Form
st.write("Enter customer details to predict if the customer will exit:")

# Create input fields
creditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1)
Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3, step=1)
Balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])
EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# Predict button
if st.button("Predict"):
    # Step 1: Prepare input data
    input_data = pd.DataFrame({
        'CreditScore': [creditScore],
        'Geography': [Geography],
        'Gender': [Gender],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })
    
    # Step 2: Encode the 'Gender' column
    input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
    
    # Step 3: Encode the 'Geography' column
    geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
    geo_columns = onehot_encoder_geo.get_feature_names_out(['Geography'])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=geo_columns)
    
    # Step 4: Combine processed features
    input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)
    
    # Step 5: Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Step 6: Predict using the model
    prediction = model.predict(scaled_input)
    
    # Display the result
    st.subheader("Prediction Result:")
    st.write(f"Exit Probability: {prediction[0][0]:.2f}")
    
    if prediction[0][0] >= 0.5:
        st.error("The customer is likely to exit.")
    else:
        st.success("The customer is not likely to exit.")
