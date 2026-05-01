import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os

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
st.write(f"Powered by Local CPU Embeddings & **Groq** (Primary: `{primary_model}` | Backup: `{backup_model}`)")

# --- Sidebar for Upload ---
with st.sidebar:
    st.header("Document Upload")
    pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type=["pdf"])

# --- Helper Functions ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    # LOCAL CPU EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
    
    # 1. Define the Primary LLM (70B Model - Smartest)
    primary_llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=primary_model, 
        temperature=0.3
    )
    
    # 2. Define the Backup LLM (8B Model - Highest Rate Limits)
    backup_llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=backup_model, 
        temperature=0.3
    )
    
    # 3. LangChain Magic: Bind them together. If primary fails, it instantly uses backup.
    llm_with_fallback = primary_llm.with_fallbacks([backup_llm])
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Pure LCEL utilizing the fallback LLM configuration
    chain = prompt | llm_with_fallback
    return chain

def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Vector DB not found. Please upload and process a document first.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # k=3 to keep chunk sizes manageable
    docs = new_db.similarity_search(user_question, k=3) 
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    
    with st.spinner("Analyzing document..."):
        try:
            response = chain.invoke({"context": context_text, "input": user_question})
            st.write("**Reply:** ", response.content) 
        except Exception as e:
            # This only triggers if BOTH the primary and the backup models fail
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
                st.sidebar.success("Done! You can now ask questions.")
    else:
        st.sidebar.error("Please upload a PDF first.")

user_question = st.text_input("Ask a question about your uploaded documents:")
if user_question:
    user_input(user_question)
