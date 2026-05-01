import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import os

hf_token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# --- Securely Load API Key (Only needed for the LLM now!) ---
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Missing GOOGLE_API_KEY. Please set it in Streamlit Secrets or environment variables.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Page Config ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Chat with your Documents (RAG)")
st.write("Upload a PDF, let the AI process it, and ask questions! Perfect for long reports, manuals, or research papers.")

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
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # LOCAL EMBEDDINGS: This runs entirely on the CPU. No API calls!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{input}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Pure LCEL
    chain = prompt | model
    return chain

def user_input(user_question):
    # FAISS safety check
    if not os.path.exists("faiss_index"):
        st.error("Vector DB not found. Please upload and process a document first.")
        return

    # LOCAL EMBEDDINGS: Must match the model used to create the DB
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the local FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    # Manually extract the text from the retrieved Document objects
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    chain = get_conversational_chain()
    
    # Invoke the pure LCEL chain
    response = chain.invoke({"context": context_text, "input": user_question})
    
    st.write("**Reply:** ", response.content) 

# --- Main App Logic ---
if st.sidebar.button("Process Documents"):
    if pdf_docs:
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            
            # Check for empty/corrupt PDFs
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
