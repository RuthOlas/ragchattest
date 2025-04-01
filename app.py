import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay
import google.generativeai as genai
import PyPDF2

# Check for SHAP library
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP library not installed. Some explanation features will be limited.")

# Load the trained model with caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model_top4.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Access the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

# Load dataset and PDF
df1 = pd.read_csv("df.csv")
pdf_path = "IFSSA and Insights.pdf"

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

pdf_text = extract_text_from_pdf(pdf_path)

# Generate AI response
def generate_response(prompt, context):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        cleaned_context = context.replace("PDF Content:", "").replace("Dataset 1 Preview:", "")
        response = model.generate_content(f"{prompt}\n\nContext:\n{cleaned_context}")
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."

# Main function
def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Predictions", "Chatbot"])

    if app_page == "Dashboard":
        st.title("Hamper Return Prediction and Chatbot App")
        st.write("Unified platform for predictions and AI-based chatbot assistance.")

    elif app_page == "Predictions":
        st.title("Hamper Return Prediction App")
        month = st.number_input("Month", min_value=1, max_value=12, step=1, value=6)
        total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1, value=5)
        avg_days_between_pickups = st.number_input("Avg Days Between Pickups", min_value=1.0, max_value=100.0, step=0.1, value=30.0)
        days_since_last_pickup = st.number_input("Days Since Last Pickup", min_value=1.0, max_value=100.0, step=0.1, value=30.0)
        input_data = {"month": month, "total_visits": total_visits, "avg_days_between_pickups": avg_days_between_pickups, "days_since_last_pickup": days_since_last_pickup}
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            if prediction[0] == 1:
                st.success(f"Likely to RETURN (probability: {probability[0][1]:.2%})")
            else:
                st.error(f"Not likely to RETURN (probability: {probability[0][0]:.2%})")

    elif app_page == "Chatbot":
        st.title("IFSSA Retention Chatbot")
        context = "\nDataset 1 Preview:\n" + df1.head(5).to_string() + "\n\nPDF Content:\n" + pdf_text[:2000]
        user_input = st.text_input("Ask a question about the data:")
        if st.button("Send"):
            response = generate_response(user_input, context)
            st.write(response)

if __name__ == "__main__":
    main()
