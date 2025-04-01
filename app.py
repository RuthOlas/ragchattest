import streamlit as st
import pandas as pd
import google.generativeai as genai

# Access the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["GOOGLE_API_KEY"]
st.write("Your API key is:", api_key)

# Configure the API key
genai.configure(api_key=api_key)

# Function to generate response from the model
def generate_response(prompt, context):
    try:
        # Initialize GenerativeModel
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')  # You can replace 'gemini-pro' with the model you want to use
        # Generate a response from the model
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text  # Use 'text' attribute for response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."

# Streamlit app
def main():
    st.title("Project-Specific Chatbot")
    st.write("Upload project-related files and ask questions based on the data.")

    # File upload
    uploaded_files = st.file_uploader("Upload your project files (CSV/Excel)", type=["csv", "xlsx"], accept_multiple_files=True)

    # Prepare data context
    dataframes = {}
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.endswith('.csv'):
                    dataframes[file.name] = pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    dataframes[file.name] = pd.read_excel(file)
                st.success(f"Successfully loaded {file.name}")
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")

    # Create context from data
    context = ""
    for file_name, df in dataframes.items():
        context += f"\nData from {file_name}:\n"
        context += df.head(5).to_string()  # Include a preview of the data (first 5 rows)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your project:", key="input")
    if st.button("Send"):
        if user_input and context:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input, context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        elif not context:
            st.error("Please upload relevant files to ask project-specific questions.")

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()
