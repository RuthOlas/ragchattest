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


# Access the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["GOOGLE_API_KEY"]

# Configure the API key
genai.configure(api_key=api_key)

# Loading dataset
df1 = pd.read_csv("df.csv")  # Update with your actual dataset file


st.markdown('<h1 style="color:green; font-size: 2.5em;">Hamper Return Prediction App</h1>', unsafe_allow_html=True)
# Set the title in green with a larger font size


# Your app logic here
st.write("Welcome to the Hamper Return Prediction App!")

# Function to extract text from a preloaded PDF
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

# Preload the PDF
pdf_path = "IFSSA and Insights.pdf"  # Update with your actual PDF file
pdf_text = extract_text_from_pdf(pdf_path)

# Function to generate response from the model
def generate_response(prompt, context):
    try:
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        
        # Remove explicit source references from context if necessary
        cleaned_context = context.replace("PDF Content:", "").replace("Dataset 1 Preview:", "")
        
        # Generate content based on the cleaned context
        response = model.generate_content(f"{prompt}\n\nContext:\n{cleaned_context}")
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."

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
        # st.success("✅ Model loaded successfully!")
        
        # Debug: Show model structure
        if st.session_state.get('debug', False):
            st.write("Model type:", type(model))
            if hasattr(model, 'named_steps'):
                st.write("Pipeline steps:", list(model.named_steps.keys()))
            if hasattr(model, 'feature_importances_'):
                st.write("Model has feature_importances_ attribute")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Define only the top 5 features
REQUIRED_COLUMNS = [
    
    "month",
    "total_visits",
    "avg_days_between_pickups",
    "days_since_last_pickup"
]

# Function to preprocess input data
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0  # Set missing columns to 0

    # Ensure the column order matches model training
    input_df = input_df[REQUIRED_COLUMNS]
    return input_df

def exploratory_data_analysis():
    st.title("Hamper Collection Insights")
    st.subheader("Power BI Visualization")
    powerbi_url = "https://app.powerbi.com/view?r=eyJrIjoiMTE4Y2JiYWQtMzNhYS00NGFiLThmMDQtMmIwMDg4YTIzMjI5IiwidCI6ImUyMjhjM2RmLTIzM2YtNDljMy05ZDc1LTFjZTI4NWI1OWM3OCJ9"
    st.components.v1.iframe(powerbi_url, width=800, height=600)

def get_model_components():
    """Extract model components handling different pipeline structures"""
    components = {
        'preprocessor': None,
        'classifier': None,
        'feature_names': REQUIRED_COLUMNS
    }
    
    if hasattr(model, 'named_steps'):
        # Handle pipeline models
        possible_preprocessor_names = ['preprocessor', 'pre', 'transform', 'features']
        possible_classifier_names = ['classifier', 'model', 'clf', 'estimator']
        
        for name in possible_preprocessor_names:
            if name in model.named_steps:
                components['preprocessor'] = model.named_steps[name]
                break
                
        for name in possible_classifier_names:
            if name in model.named_steps:
                components['classifier'] = model.named_steps[name]
                break
    else:
        # Handle non-pipeline models
        components['classifier'] = model
    
    # Get feature names if preprocessor exists
    if components['preprocessor'] is not None and hasattr(components['preprocessor'], 'get_feature_names_out'):
        components['feature_names'] = [name.split('__')[-1] for name in components['preprocessor'].get_feature_names_out()]
    
    return components

def show_shap_analysis(input_df, prediction, probability):
    if not SHAP_AVAILABLE:
        st.warning("""
        **SHAP explanations unavailable**  
        To enable full explanation features, please install SHAP:  
        `pip install shap` or restart the app in an environment with SHAP installed.
        """)
        return
        
    st.subheader("🔍 Prediction Explanation")
    
    try:
        components = get_model_components()
        classifier = components['classifier']
        
        if classifier is None:
            st.error("Could not identify classifier in model pipeline")
            return
            
        # Process input data
        if components['preprocessor'] is not None:
            X_processed = components['preprocessor'].transform(input_df)
        else:
            X_processed = input_df.values
            
        feature_names = components['feature_names']
        
        # Convert to dense if sparse
        if hasattr(X_processed, 'toarray'):
            X_processed = X_processed.toarray()
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_processed)
        
        # For binary classification
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Feature Importance", "Prediction Breakdown"])
        
        with tab1:
            st.write("**Global Feature Importance**")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X_processed, feature_names=feature_names, plot_type="bar", show=False)
            plt.tight_layout()
            st.pyplot(fig)
            st.caption("This shows which features most influence model predictions overall.")
        
        with tab2:
            st.write("**How Each Feature Contributed to This Prediction**")
            
            # Create waterfall plot
            base_value = explainer.expected_value
            contribution = shap_values[0]
            
            # Create dataframe for plot
            df_waterfall = pd.DataFrame({
                'Feature': ['Base Value'] + feature_names,
                'SHAP Value': [base_value] + list(contribution)
            })
            
            # Color based on positive/negative contribution
            df_waterfall['Color'] = df_waterfall['SHAP Value'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative')
            
            fig = px.bar(df_waterfall, 
                        x='SHAP Value', 
                        y='Feature', 
                        color='Color',
                        color_discrete_map={'Positive': '#FF0051', 'Negative': '#008BFB'},
                        orientation='h',
                        title=f"Feature Contributions to Prediction")
            
            # Add base value line
            fig.add_vline(x=base_value, line_dash="dash", line_color="gray")
            fig.update_layout(showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature values
            st.write("**Feature Values for This Prediction:**")
            feature_values = {f: v for f, v in zip(feature_names, X_processed[0])}
            st.json(feature_values)
    
    except Exception as e:
        st.error(f"Error generating explanations: {str(e)}")
        st.warning("Model structure may not be compatible with automatic explanation generation")

def show_confidence_analysis(probability):
    st.subheader("📊 Prediction Confidence")
    
    # Create confidence gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability[0][1],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level"},
        gauge = {
            'axis': {'range': [0, 1]},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgray"},
                {'range': [0.3, 0.7], 'color': "gray"},
                {'range': [0.7, 1], 'color': "darkgray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability[0][1]}}
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    if probability[0][1] < 0.3 or probability[0][1] > 0.7:
        st.success("✅ High confidence prediction")
        st.write("The model is very confident in this prediction due to clear patterns in the input data.")
    else:
        st.warning("⚠️ Medium confidence prediction")
        st.write("The model has moderate confidence. Consider verifying with additional client information.")

def predictions_page():
    st.title("Hamper Return Prediction App")
    st.write("Enter details to predict if a client will return.")
    
    # User input fields
    col1, col2 = st.columns(2)
    with col1:
        month = st.number_input("Month", min_value=1, max_value=12, step=1, value=6)
        total_visits = st.number_input("Total Visits", min_value=1, max_value=100, step=1, value=5)
        
    with col2:
        avg_days_between_pickups = st.number_input("Avg Days Between Pickups", 
                                                min_value=1.0, max_value=100.0, 
                                                step=0.1, value=30.0)
        days_since_last_pickup = st.number_input("Days Since Last Pickup", 
                                                min_value=1.0, max_value=100.0, 
                                                step=0.1, value=30.0)
    
    input_data = {
        "month": month,
        "total_visits": total_visits,
        "avg_days_between_pickups": avg_days_between_pickups,
        "days_since_last_pickup": days_since_last_pickup
        
    }
    
    # Prediction button
    if st.button("Predict"):
        if model is None:
            st.error("Model not loaded. Please check if 'model_top4.pkl' exists.")
        else:
            input_df = preprocess_input(input_data)
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            # Show prediction result
            st.subheader("🎯 Prediction Result")
            
            if prediction[0] == 1:
                st.success(f"✅ The client is **likely to RETURN** (probability: {probability[0][1]:.2%})")
            else:
                st.error(f"❌ The client is **NOT likely to RETURN** (probability: {probability[0][0]:.2%})")
            
            # Show SHAP analysis
            show_shap_analysis(input_df, prediction, probability)
            
            # Show confidence analysis
            show_confidence_analysis(probability)
            
            # Recommendation based on prediction
            st.subheader("💡 Recommended Actions")
            if prediction[0] == 1:
                st.markdown("""
                - **Follow up:** Schedule a reminder for their next expected pickup
                - **Retention:** Consider adding them to loyalty communications
                - **Feedback:** Ask about their experience to maintain engagement
                """)
            else:
                st.markdown("""
                - **Outreach:** Consider a check-in call to understand their situation
                - **Incentives:** Offer additional support if appropriate
                - **Feedback:** Learn why they might not be returning
                """)
def chatbox():
    # Function to extract text from a preloaded PDF
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

    # Preload the PDF
    pdf_path = "IFSSA and Insights.pdf"  # Update with your actual PDF file
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        pdf_text = ""  # Ensure pdf_text is an empty string if extraction fails

    # Function to generate response from the model
    def generate_response(prompt, context):
        try:
            model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
            # Remove explicit source references from context if necessary
            cleaned_context = context.replace("PDF Content:", "").replace("Dataset 1 Preview:", "")
            # Generate content based on the cleaned context
            response = model.generate_content(f"{prompt}\n\nContext:\n{cleaned_context}")
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request."

    st.markdown("""
        <style>
        .stApp {
            background-color: #ffffff;
            color: #333333;
        }
        .stTextInput, .stButton {
            background-color: #4caf50;
            color: white;
        }
        .stTitle, .stHeader, .stText {
            color: #4caf50;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("IFSSA Retention Chatbot")
    st.write("Ask questions based on your datasets.")

    # Create context from dataset
    context = "\nDataset 1 Preview:\n" + df1.head(5).to_string()
    context += "\n\nPDF Content:\n" + pdf_text[:2000]  # Limit text for efficiency

    # Initialize chat history if not already initialized
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_input = st.text_input("Ask a question about our data:", key="input")

    # If user presses "Send", process the question
    if st.button("Send"):
        if user_input:
            # Store user message at the top of the chat history
            st.session_state.chat_history.insert(0, {"role": "user", "content": user_input})
            # Generate response from the model
            response = generate_response(user_input, context)
            # Store assistant response below the user message
            st.session_state.chat_history.insert(1, {"role": "assistant", "content": response})

    # Display chat history, with most recent messages on top
    for message in st.session_state.chat_history:
        st.write(f"**{message['role'].capitalize()}**: {message['content']}")

def dashboard():
    header_image_url = "https://raw.githubusercontent.com/ChiomaUU/Client-Prediction/refs/heads/main/ifssa_2844cc71-4dca-48ae-93c6-43295187e7ca.avif"
    st.image(header_image_url, use_container_width=True)

    #st.title("Hamper Return Prediction App")
    st.write("This app predicts whether a client will return for food hampers using machine learning.")
    
    st.markdown("""
    ### Key Features:
    - **Predictions:** Estimate return probability for individual clients
    - **Explainability:** Understand why the model makes each prediction
    - **Confidence Metrics:** See how certain the model is about each prediction
    """)
    
    st.markdown("""
    ### How It Works:
    1. Navigate to the **Predictions** page
    2. Enter client details
    3. Get the prediction with explanations
    4. View recommended actions
    """)

def main():
    st.sidebar.title("Navigation")
    app_page = st.sidebar.radio("Choose a page", ["Dashboard", "Insights", "Predictions", "Chatbox"])
    
    # Debug toggle
    if st.sidebar.checkbox("Debug mode"):
        st.session_state.debug = True
    else:
        st.session_state.debug = False

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "Insights":
        exploratory_data_analysis()
    elif app_page == "Predictions":
        predictions_page()
    elif app_page == "Chatbox":
        chatbox()

    # Add footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About This App:**
    - Uses machine learning model
    - Provides explainable AI insights
    - Designed for food bank client retention
    """)

if __name__ == "__main__":
    main()
