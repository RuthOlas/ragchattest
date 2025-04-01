import streamlit as st
import pandas as pd
import google.generativeai as genai

# Access the API key from Streamlit secrets
api_key = st.secrets["api_keys"]["GOOGLE_API_KEY"]
#st.write("Your API key is:", api_key)

# Configure the API key
genai.configure(api_key=api_key)

# Loading dataset 
df1 = pd.read_csv("df.csv")

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
        response = model.generate_content(f"{prompt}\n\nContext:\n{context}")
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process your request."

# Streamlit app
def main():
    st.title("IFSA Retention Chatbot")
    st.write("Ask questions based on your datasets.")

    # Create context from dataset
    context = "\nDataset 1 Preview:\n" + df1.head(5).to_string()
    context += "\n\nPDF Content:\n" + pdf_text[:2000]  # Limit text for efficiency
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about our data:", key="input")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.insert(0, {"role": "user", "content": user_input})
            response = generate_response(user_input, context)
            st.session_state.chat_history.insert(1, {"role": "assistant", "content": response})

    for message in st.session_state.chat_history:
        st.write(f"{message['role'].capitalize()}: {message['content']}")

if __name__ == "__main__":
    main()

