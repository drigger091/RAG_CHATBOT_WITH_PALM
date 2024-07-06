import streamlit as st
import time
from src.helper import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain

def user_input(user_question):
    if not st.session_state.chatHistory:
        st.session_state.chatHistory = []
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("User:", message.content)
        else:
            st.write("Bot:", message.content)

def main():
    st.set_page_config(
        page_title="Rag chatbot testing",
        page_icon="🤖",
        layout="wide"
    )
    st.header("Retrieval System Implementation")

    user_question = st.text_input("Ask Me a Question from your PDF")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    if user_question:
        if st.session_state.conversation is not None:
            user_input(user_question)
        else:
            st.warning("Please upload and process a PDF file first.")

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload your PDF file and click on submit and process button", type=['pdf'])

        if st.button("Submit and Process"):
            if pdf_docs is not None:
                with st.spinner('I am reading Yeah :D.....'):
                    # Save the uploaded file temporarily to read it
                    with open("temp.pdf", "wb") as f:
                        f.write(pdf_docs.read())
                    
                    # Initialize the Streamlit progress bar
                    progress_bar = st.progress(0)
                    
                    # Update the progress bar at each step
                    raw_text = get_pdf_text("temp.pdf")
                    progress_bar.progress(20)
                    chunks = get_text_chunks(raw_text)
                    progress_bar.progress(40)
                    vector_store = get_vector_store(chunks)
                    progress_bar.progress(80)
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    progress_bar.progress(100)

                    st.success("Done")

if __name__ == "__main__":
    main()
