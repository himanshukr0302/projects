# 1st Version: 2025-08-09
"""
📚 AI Study Buddy
Upload your syllabus/book PDF and ask questions.
Default: Gemini (Google Generative AI) + FAISS + RAG
Optional: OpenAI or HuggingFace Embeddings (see commented code)
"""

# ===== Imports =====
import os
import tempfile
import streamlit as st
import pypdf
from dotenv import load_dotenv

# LangChain core tools
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

# ===== LLM & Embeddings =====
# --- Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# --- HuggingFace Embeddings (Offline, free) ---
# from langchain_community.embeddings import HuggingFaceEmbeddings

# --- OpenAI (Paid API) ---
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ===== Config & API Keys =====
load_dotenv()

# Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("Please set your GEMINI_API_KEY in a .env file")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# OpenAI settings (if switching)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# if not OPENAI_API_KEY:
#     st.error("Please set your OPENAI_API_KEY in a .env file")
#     st.stop()

# Embedding & Model configs
GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_CHAT_MODEL = "gemini-2.5-pro"
INDEX_PATH = "faiss_index"


# ===== Utility Functions =====
def extract_text_from_pdf(file_path):
    """Extract text from PDF file."""
    reader = pypdf.PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def create_vector_store(text):
    """Create FAISS vector store from raw text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
    docs = splitter.split_text(text)

    # --- Choose embeddings ---
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)  # Default: Gemini
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Local
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)  # OpenAI

    vectordb = FAISS.from_texts(docs, embedding=embeddings)
    vectordb.save_local(INDEX_PATH)
    return vectordb


def load_vector_store():
    """Load FAISS vector store from disk."""
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBED_MODEL)  # Default: Gemini
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_qa_chain(vectordb):
    """Create a Retrieval-QA chain."""
    llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)  # Default: Gemini
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)  # OpenAI

    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


# ===== Streamlit UI =====
st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy")
st.markdown("Upload your syllabus/book PDF and ask questions. Powered by LLM + RAG.")

# --- Sidebar: PDF upload ---
with st.sidebar:
    st.header("📄 Upload Syllabus")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            text = extract_text_from_pdf(tmp_file.name)
            create_vector_store(text)
        st.success("✅ Syllabus indexed successfully!")

# --- Chat Interface ---
if os.path.exists(INDEX_PATH):
    vectordb = load_vector_store()
    qa_chain = get_qa_chain(vectordb)

    if "history" not in st.session_state:
        st.session_state.history = []

    user_input = st.text_input("Ask a question from the syllabus:", "")
    if user_input:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_input)
        st.session_state.history.append(("You", user_input))
        st.session_state.history.append(("AI", answer))

    # Display conversation history
    for speaker, text in st.session_state.history:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {text}")
        else:
            st.markdown(f"**🤖 AI:** {text}")

else:
    st.info("Please upload a syllabus PDF first.")
