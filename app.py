import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with your Documents (RAG)")
st.write("Upload a PDF, let the AI process it, and ask questions! Perfect for long reports, manuals, or research papers.")

# --- Sidebar for API Key & Upload ---
with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    st.markdown("[Get your free API key here](https://aistudio.google.com/app/apikey)")
    
    st.divider()
    st.header("Document Upload")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Load the local FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("**Reply:** ", response["output_text"])

# --- Main App Logic ---
if not api_key:
    st.warning("Please enter your API Key in the sidebar to proceed.")
else:
    if st.sidebar.button("Process Documents"):
        if pdf_docs:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.sidebar.success("Done! You can now ask questions.")
        else:
            st.sidebar.error("Please upload a PDF first.")

    user_question = st.text_input("Ask a question about your uploaded documents:")
    if user_question:
        user_input(user_question, api_key)
