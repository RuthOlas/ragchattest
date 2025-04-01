import streamlit as st
import pandas as pd
import google.generativeai as genai

# Access the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["GOOGLE_API_KEY"]
#st.write("Your API key is:", api_key)

# Configure the API key
genai.configure(api_key=api_key)

# Load your dataset (Replace 'your_dataframe' with your actual DataFrame)
df1 = pd.read_csv("Food Hampers.csv")
df2 = pd.read_csv("Clients Data Dimension.csv")

# Function to generate response from the model
def generate_response(prompt, context):
    try:
        # Initialize GenerativeModel
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')  
        # Generate a response from the model
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text  # Use 'text' attribute for response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."

# Streamlit app
def main():
    st.title("IFSSA Retention Chatbot")
    st.write("Ask questions based on our datasets.")

    # Create context from your datasets
    context = "\nDataset 1 Preview:\n" + df1.head(5).to_string()
    context += "\n\nDataset 2 Preview:\n" + df2.head(5).to_string()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about your data:", key="input")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input, context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()

