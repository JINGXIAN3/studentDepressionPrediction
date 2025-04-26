import requests
import pickle
from io import BytesIO
import streamlit as st

def load_model_from_github(model_url):
    try:
        # Make a GET request to the URL
        response = requests.get(model_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Try to load the pickle model from the byte content
            model = pickle.load(BytesIO(response.content))
            return model
        else:
            raise Exception(f"Failed to fetch model, status code: {response.status_code}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define the model URL (replace with the correct URL)
model_url = "https://raw.githubusercontent.com/yourusername/yourrepo/main/logistic_model.pkl"

# Load the model
logistic_model = load_model_from_github(model_url)

if logistic_model is not None:
    st.title("Depression Prediction App")
    
    # Input fields for user inputs
    age = st.number_input('Age', min_value=18, max_value=34)
    study_satisfaction = st.slider('Study Satisfaction (0-5)', 0, 5)
    dietary_habits = st.slider('Dietary Habits (0-3)', 0, 3)
    suicidal_thoughts = st.selectbox('Suicidal Thoughts (Yes=1, No=0)', [0, 1])
    work_study_hours = st.slider('Work/Study Hours', 0, 12)
    overall_stress = st.slider('Overall Stress (0-9)', 0, 9)
    
    if st.button('Predict Depression'):
        input_data = np.array([[age, study_satisfaction, dietary_habits, suicidal_thoughts, work_study_hours, overall_stress]])
        prediction = logistic_model.predict(input_data)
        
        if prediction[0] == 1:
            st.error('Prediction: Likely Depression')
        else:
            st.success('Prediction: No Depression')
else:
    st.error("Model not loaded properly.")
