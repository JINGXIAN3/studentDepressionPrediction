import streamlit as st
import joblib 
import numpy as np
import os

# Load model locally
logistic_model = joblib.load('logistic_model.joblib')  # <-- joblib load

st.title("Depression Prediction App")

# Input fields
age = st.number_input('Age', min_value=18, max_value=34)
study_satisfaction = st.slider('Study Satisfaction (0-5)', 0, 5)
dietary_habits = st.slider('Dietary Habits (0-3)', 0, 3)
suicidal_thoughts = st.selectbox('Suicidal Thoughts (Yes=1, No=0)', [0, 1])
work_study_hours = st.slider('Work/Study Hours', 0, 12)
overall_stress = st.slider('Overall Stress (0-9)', 0, 9)

# Predict button
if st.button('Predict Depression'):
    input_data = np.array([[age, study_satisfaction, dietary_habits, suicidal_thoughts, work_study_hours, overall_stress]])
    prediction = logistic_model.predict(input_data)

    if prediction[0] == 1:
        st.error('Prediction: Likely Depression')
    else:
        st.success('Prediction: No Depression')
