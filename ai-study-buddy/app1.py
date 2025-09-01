# 2st Version: 2025-08-10
"""
app.py - AI Study Buddy (Streamlit)
Features:
- Upload PDF -> index with HuggingFace embeddings -> FAISS
- RAG QA using chosen LLM: OpenAI or Google Gemini
- Generate practice quizzes (MCQs)
- Optional voice input (speech_recognition) and TTS (gTTS)
- Optional translation via transformers (Helsinki)
"""

import os
import tempfile
import time
import json
import streamlit as st
import pypdf
from dotenv import load_dotenv

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLMs (imports present and ready; commented alternatives shown)
from langchain.chat_models import ChatOpenAI  # OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini

# Optional: OpenAI embeddings if you ever want to use them (commented)
# from langchain.embeddings import OpenAIEmbeddings

# Optional translator (Helsinki models) - may be slow & large
try:
    from transformers import pipeline as hf_pipeline
    translator_en_to_hi = hf_pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    translator_hi_to_en = hf_pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-hi-en")
except Exception:
    translator_en_to_hi = None
    translator_hi_to_en = None

# STT/TTS optional libs (may not be available in all environments)
try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

# ---- Config ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Index folder
INDEX_PATH = "faiss_index"

# Embedding model (HuggingFace local)
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Chat models
OPENAI_CHAT_MODEL = "gpt-4o-mini"  # or "gpt-4o"
GEMINI_CHAT_MODEL = "gemini-2.5-pro"

# ---- Helper utils ----
def extract_text_from_pdf(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return "\n".join(pages)

@st.cache_data(show_spinner=False)
def create_vector_store_from_text(text: str, index_path=INDEX_PATH, chunk_size=1500, chunk_overlap=100):
    """Create FAISS index using HuggingFace embeddings (cached)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectordb = FAISS.from_texts(docs, embedding=embeddings)
    vectordb.save_local(index_path)
    return vectordb

def load_vector_store(index_path=INDEX_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def build_qa_chain(vectordb, llm_choice="Gemini"):
    if llm_choice == "OpenAI":
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not found in environment.")
            st.stop()
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    else:
        # Gemini
        if not GEMINI_API_KEY:
            st.warning("GEMINI_API_KEY not found: Gemini will not work until you set it.")
        llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def speak_text(text: str, lang="en"):
    """Generate TTS and return audio file path (mp3)."""
    if gTTS is None:
        st.warning("gTTS not installed — TTS not available.")
        return None
    t = gTTS(text=text, lang="hi" if lang.startswith("hi") else "en")
    out = "tts_output.mp3"
    t.save(out)
    return out

def recognize_speech_from_mic(lang_code="en-IN", timeout=5):
    """Record from microphone and return recognized text. Requires speech_recognition and a mic."""
    if sr is None:
        st.warning("speech_recognition not installed — voice input unavailable.")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... speak now")
        audio = r.listen(source, timeout=timeout)
    try:
        return r.recognize_google(audio, language=lang_code)
    except Exception as e:
        st.warning(f"STT failed: {e}")
        return ""

def translate_text(text: str, target_lang: str):
    """Translate between English and Hindi with transformers if available."""
    if translator_en_to_hi is None:
        st.info("Transformers translator not available; skipping translation.")
        return text
    if target_lang.startswith("hi"):
        return translator_en_to_hi(text)[0]["translation_text"]
    else:
        return translator_hi_to_en(text)[0]["translation_text"]

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy")
st.markdown("Upload your syllabus/book PDF and ask questions. Powered by LLM + RAG.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    llm_choice = st.radio("LLM:", ["Gemini", "OpenAI"])
    language = st.selectbox("Answer language:", ["Auto (detect)", "English", "Hindi"])
    use_voice_in = st.checkbox("Enable voice input (mic)", value=False)
    use_voice_out = st.checkbox("Enable voice output (TTS)", value=False)
    st.markdown("---")
    st.caption("Notes: Embeddings use local HuggingFace model (no quota).")

# Upload PDF
uploaded_file = st.sidebar.file_uploader("Upload syllabus PDF", type=["pdf"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    st.sidebar.success("Saved uploaded PDF")
    st.sidebar.info("Indexing... (this may take a moment on first run)")

    text = extract_text_from_pdf(tmp_path)
    # Create index (cached by Streamlit)
    with st.spinner("Creating vector index (HuggingFace embeddings + FAISS)..."):
        create_vector_store_from_text(text, index_path=INDEX_PATH)
    st.sidebar.success("Indexed!")

# If index exists, load and create chain
if os.path.exists(INDEX_PATH):
    vectordb = load_vector_store(INDEX_PATH)
    qa_chain = build_qa_chain(vectordb, llm_choice=llm_choice)
else:
    vectordb = None
    qa_chain = None

# Main area: Chat and Quiz
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat / Ask questions")
    if qa_chain is None:
        st.info("Please upload and index a PDF from the sidebar first.")
    else:
        # Voice input
        user_query = ""
        if use_voice_in:
            if st.button("Record voice query"):
                # Try STT
                lang_code = "hi-IN" if language.startswith("Hindi") else "en-IN"
                res = recognize_speech_from_mic(lang_code=lang_code)
                if res:
                    user_query = res
                    st.write("You said:", res)
        # Text input
        typed = st.text_input("Or type your question here:")
        if typed:
            user_query = typed

        if st.button("Get Answer") and user_query:
            with st.spinner("Retrieving relevant docs and generating answer..."):
                answer = qa_chain.run(user_query)

            # Translation if needed
            if language == "Hindi":
                # translate english->hindi
                answer_out = translate_text(answer, target_lang="hi")
            elif language == "English":
                answer_out = answer
            else:
                # Auto: keep as is
                answer_out = answer

            st.markdown("**Answer:**")
            st.write(answer_out)

            # TTS
            if use_voice_out:
                audio_file = speak_text(answer_out, lang="hi" if language.startswith("Hindi") else "en")
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")

with col2:
    st.subheader("Quiz Generator")
    if vectordb is None:
        st.info("Upload and index a PDF to enable quiz generation.")
    else:
        topic = st.text_input("Topic / chapter (leave blank to use whole doc):")
        num_q = st.slider("Number of questions", 1, 10, 5)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        if st.button("Generate Quiz"):
            # Fetch context: we retrieve top docs for topic or default
            query_for_context = topic if topic.strip() else "Summarize main points"
            docs = vectordb.similarity_search(query_for_context, k=6)
            context_text = "\n".join([d.page_content for d in docs])

            quiz_prompt = f"""
You are an expert teacher. From the following context, generate {num_q} multiple-choice questions (4 options each), indicate the correct option,
and provide a short explanation for the correct answer. Difficulty: {difficulty}.
Context:
{context_text}
"""
            # Use LLM (same setting as chat)
            if llm_choice == "OpenAI":
                llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
            else:
                llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL)

            # We can call the model directly; use simple chat interface via .generate or .generate? For LangChain chat_models: use .generate / .predict
            try:
                quiz_response = llm.predict(quiz_prompt)  # ChatOpenAI/ChatGoogleGenerativeAI support .predict
            except Exception:
                # fallback to run via simple wrapper
                quiz_response = llm.predict(quiz_prompt)

            st.markdown("### Quiz")
            st.write(quiz_response)

            # Save quiz to session for answering
            if "last_quiz" not in st.session_state:
                st.session_state.last_quiz = ""
            st.session_state.last_quiz = quiz_response

        # If quiz present, show simple "take quiz" area
        if "last_quiz" in st.session_state and st.session_state.last_quiz:
            st.markdown("#### Last generated quiz (you can copy/paste answers)")
            st.text_area("Quiz", st.session_state.last_quiz, height=250)

# Footer / tips
st.markdown("---")
st.markdown("**Tips:** If you plan to run large PDFs, first index once and reuse the `faiss_index` folder. "
            "If embeddings are slow, consider running with a GPU or using a smaller HF model.")

# ---- End of file ----
