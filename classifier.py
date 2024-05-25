import streamlit as st
import pandas as pd
import joblib

# Load the test data
test_data = pd.read_csv(r"D:\Internship tasks\Task1and2\test.csv")

# Load the trained models
model_files = {
    'Logistic Regression': r'D:\Internship tasks\Task1and2\logistic_regression_model.pkl',
    'Decision Tree': r'D:\Internship tasks\Task1and2\decision_tree_model.pkl',
    'Random Forest': r'D:\Internship tasks\Task1and2\random_forest_model.pkl',
    'SVM': r'D:\Internship tasks\Task1and2\svm_model.pkl'
}

# Function to make predictions
def predict(model_name, data):
    model = joblib.load(model_files[model_name])
    predictions = model.predict(data)
    return predictions

# Streamlit UI
st.title('Model Prediction')

# Display the test data
st.write('Test Data:')
st.write(test_data)

# Input box for user to enter data
user_input = st.text_input('Enter your data separated by spaces:')
if user_input:
    user_data = [float(val) for val in user_input.split()]
    user_data = [user_data]  # Convert to 2D array
    selected_model = st.selectbox('Select a model:', list(model_files.keys()))
    if st.button('Predict'):
        predictions = predict(selected_model, user_data)
        st.write('Predictions:')
        st.write(predictions)