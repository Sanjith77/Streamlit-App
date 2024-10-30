import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('titanic_logistic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Title and description
st.title("Titanic Survival Prediction")
st.write("Predict the survival of Titanic passengers based on specific characteristics.")

# Input fields for user data
pclass = st.selectbox("Passenger Class", options=[1, 2, 3])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
sex_male = st.selectbox("Sex", options=["Female", "Male"])
embarked = st.selectbox("Port of Embarkation", options=["C", "Q", "S"])

# Convert inputs into model format
sex_male = 1 if sex_male == "Male" else 0
embarked_q, embarked_s = (1, 0) if embarked == "Q" else (0, 1) if embarked == "S" else (0, 0)

user_inputs = np.array([[pclass, age, sibsp, parch, fare, sex_male, embarked_q, embarked_s]])

# Prediction
prediction = model.predict(user_inputs)

# Display prediction
st.write("Prediction (1 = Survived, 0 = Did Not Survive):", prediction[0])

