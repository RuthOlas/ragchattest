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
