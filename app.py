import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os
import shutil

# --- Securely Load API Key & Model Settings ---
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please set it in Streamlit Secrets or environment variables.")
    st.stop()

# Load Primary and Backup models from secrets (with safe defaults)
primary_model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
backup_model = st.secrets.get("GROQ_BACKUP_MODEL", "llama-3.1-8b-instant")

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with your Documents (RAG)")
st.write("An enterprise-grade Retrieval-Augmented Generation (RAG) system. Securely query, summarize, and extract insights from your documents using local vector search and high-speed AI inference.")

# --- Sidebar for Upload ---
with st.sidebar:
    st.header("Document Upload")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])
    
    st.divider()
    st.header("Database Management")
    # NEW: Added a way to clear the database so it doesn't get infinitely large
    if st.button("Clear Database 🗑️"):
        if os.path.exists("faiss_index"):
            shutil.rmtree("faiss_index")
            st.success("Vector database successfully wiped!")
        else:
            st.info("Database is already empty.")

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                # FIXED: Added a space to prevent words from sticking together across pages
                text += page_text + "\n" 
        # FIXED: Added a distinct separator between multiple PDFs so the AI doesn't mix them up
        text += "\n\n--- NEW DOCUMENT ---\n\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # FIXED: The Incremental Storage Logic
    if os.path.exists("faiss_index"):
        # If DB exists, load it and APPEND the new documents
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        vector_store.add_texts(text_chunks)
    else:
        # If DB does not exist, CREATE a new one
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{input}\n
    Answer:
    """
    
    primary_llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=primary_model, 
        temperature=0.3
    )
    
    backup_llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=backup_model, 
        temperature=0.3
    )
    
    llm_with_fallback = primary_llm.with_fallbacks([backup_llm])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    chain = prompt | llm_with_fallback
    return chain

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Vector DB not found. Please upload and process a document first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question, k=3) 
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    
    with st.spinner("Analyzing document..."):
        try:
            response = chain.invoke({"context": context_text, "input": user_question})
            st.write("**Reply:** ", response.content) 
        except Exception as e:
            st.error(f"Both primary and backup models failed. Error: {str(e)}")

# --- Main App Logic ---
if st.sidebar.button("Process Documents"):
    if pdf_docs:
        with st.spinner("Vectorizing locally..."):
            raw_text = get_pdf_text(pdf_docs)
            
            if not raw_text.strip():
                st.sidebar.error("No readable text found in the PDF. It might be an image or corrupt.")
            else:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.sidebar.success("Done! You can now ask questions about all uploaded files.")
    else:
        st.sidebar.error("Please upload a PDF first.")

user_question = st.text_input("Ask a question about your uploaded documents:")
if user_question:
    user_input(user_question)
