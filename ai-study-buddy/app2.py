# 3st Version: 2025-08-13
"""
app.py - AI Study Buddy (Streamlit) + SQLSaver (enhanced with safe JSON parsing)
Stores Q&A, quizzes, quiz attempts, and PDF chunks+embeddings in SQLite using a SQLSaver adapter.
- Works across restarts (persistent studybuddy.db)
- Multi-user support via user_id
- Shows past Q&A, quizzes, and quiz results at bottom
- Quiz generation outputs structured JSON; in-app attempt UI parses, scores, and saves results
"""

import os
import tempfile
import time
import json
import pickle
import sqlite3
import re
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pypdf
from dotenv import load_dotenv

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# LLMs
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional translator
try:
    from transformers import pipeline as hf_pipeline
    translator_en_to_hi = hf_pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
    translator_hi_to_en = hf_pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-hi-en")
except Exception:
    translator_en_to_hi = None
    translator_hi_to_en = None

# Optional STT/TTS
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

INDEX_PATH = "faiss_index"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
GEMINI_CHAT_MODEL = "gemini-2.5-pro"
DB_PATH = "studybuddy.db"

# ==========================
# Safe JSON parsing helper
# ==========================
def safe_json_loads(data):
    """Safely parse JSON from possibly messy LLM string output."""
    if isinstance(data, dict):
        return data  # Already parsed
    
    if not isinstance(data, str):
        return {}

    try:
        # Remove all code fences and extra backticks
        cleaned = re.sub(r"```(?:json)?", "", data, flags=re.IGNORECASE).strip("` \n\r\t")
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"[Warning] JSON parsing failed: {e}")
        return {}

# ==========================
# SQLite-backed SQLSaver shim
# ==========================
class SQLSaver:
    """A lightweight adapter inspired by LangGraph's SQLSaver.
    Provides simple save/load plus helpers for documents, quizzes, and results.
    """
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        # Generic events/items table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                user_id TEXT DEFAULT 'guest',
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Documents table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                doc_id TEXT,
                chunk_id INTEGER,
                content TEXT,
                embedding BLOB
            )
            """
        )
        # Quizzes table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                topic TEXT,
                difficulty TEXT,
                quiz_json TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        # Quiz results table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'guest',
                quiz_id INTEGER,
                answers_json TEXT,
                score INTEGER,
                total INTEGER,
                created_at REAL NOT NULL,
                FOREIGN KEY(quiz_id) REFERENCES quizzes(id)
            )
            """
        )
        self.conn.commit()

    # ----- Generic save/load API -----
    def save(self, obj: Dict[str, Any], user_id: str = "guest"):
        payload = json.dumps(obj)
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO items(type, user_id, payload, created_at) VALUES (?, ?, ?, ?)",
            (obj.get("type", "unknown"), user_id, payload, time.time()),
        )
        self.conn.commit()

    def load_all(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if user_id:
            cur.execute("SELECT payload FROM items WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        else:
            cur.execute("SELECT payload FROM items ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [safe_json_loads(r[0]) for r in rows]

    # ----- Documents helpers -----
    def save_document_chunks(self, user_id: str, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
        cur = self.conn.cursor()
        for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO documents(user_id, doc_id, chunk_id, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (user_id, doc_id, idx, text, pickle.dumps(emb)),
            )
        self.conn.commit()

    def list_documents(self, user_id: Optional[str] = None) -> List[Tuple]:
        cur = self.conn.cursor()
        if user_id:
            cur.execute("SELECT DISTINCT doc_id FROM documents WHERE user_id=?", (user_id,))
        else:
            cur.execute("SELECT DISTINCT doc_id FROM documents")
        return [r[0] for r in cur.fetchall()]

    # ----- Quizzes helpers -----
    def save_quiz(self, user_id: str, topic: str, difficulty: str, quiz_json: Dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO quizzes(user_id, topic, difficulty, quiz_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, topic, difficulty, json.dumps(quiz_json), time.time()),
        )
        quiz_id = cur.lastrowid
        self.conn.commit()
        return quiz_id

    def list_quizzes(self, user_id: Optional[str] = None) -> List[Tuple[int, str, str, Dict[str, Any], float]]:
        cur = self.conn.cursor()
        if user_id:
            cur.execute("SELECT id, topic, difficulty, quiz_json, created_at FROM quizzes WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        else:
            cur.execute("SELECT id, topic, difficulty, quiz_json, created_at FROM quizzes ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [(r[0], r[1], r[2], safe_json_loads(r[3]), r[4]) for r in rows]

    def get_quiz(self, quiz_id: int) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT quiz_json FROM quizzes WHERE id=?", (quiz_id,))
        row = cur.fetchone()
        return safe_json_loads(row[0]) if row else None

    def save_quiz_result(self, user_id: str, quiz_id: int, answers: Dict[str, Any], score: int, total: int) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO quiz_results(user_id, quiz_id, answers_json, score, total, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, quiz_id, json.dumps(answers), score, total, time.time()),
        )
        rid = cur.lastrowid
        self.conn.commit()
        return rid

    def list_quiz_results(self, user_id: Optional[str] = None) -> List[Tuple]:
        cur = self.conn.cursor()
        if user_id:
            cur.execute(
                "SELECT id, quiz_id, answers_json, score, total, created_at FROM quiz_results WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            )
        else:
            cur.execute(
                "SELECT id, quiz_id, answers_json, score, total, created_at FROM quiz_results ORDER BY created_at DESC"
            )
        rows = cur.fetchall()
        return [(r[0], r[1], safe_json_loads(r[2]), r[3], r[4], r[5]) for r in rows]

# ---- Database Setup ----
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
sql_saver = SQLSaver(conn)

# ---- Helper utils ----
def extract_text_from_pdf(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

@st.cache_data(show_spinner=False)
def create_vector_store_from_text(text: str, index_path=INDEX_PATH, chunk_size=1500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectordb = FAISS.from_texts(docs, embedding=embeddings)
    vectordb.save_local(index_path)
    return vectordb, docs, embeddings

@st.cache_data(show_spinner=False)
def load_vector_store(index_path=INDEX_PATH):
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

def build_qa_chain(vectordb, llm_choice="Gemini"):
    if llm_choice == "OpenAI":
        if not OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not found.")
            st.stop()
        llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    else:
        if not GEMINI_API_KEY:
            st.warning("GEMINI_API_KEY not found.")
        llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def speak_text(text: str, lang="en"):
    if gTTS is None:
        st.warning("gTTS not installed.")
        return None
    t = gTTS(text=text, lang="hi" if lang.startswith("hi") else "en")
    out = "tts_output.mp3"
    t.save(out)
    return out

def recognize_speech_from_mic(lang_code="en-IN", timeout=5):
    if sr is None:
        st.warning("speech_recognition not installed.")
        return ""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source, timeout=timeout)
    try:
        return r.recognize_google(audio, language=lang_code)
    except Exception as e:
        st.warning(f"STT failed: {e}")
        return ""

def translate_text(text: str, target_lang: str):
    if translator_en_to_hi is None:
        return text
    if target_lang.startswith("hi"):
        return translator_en_to_hi(text)[0]["translation_text"]
    else:
        return translator_hi_to_en(text)[0]["translation_text"]

# ---------- Quiz helpers ----------
QUIZ_JSON_INSTRUCTIONS = (
    "Return a strict JSON object with this schema: {\n"
    "  \"questions\": [\n"
    "    {\n"
    "      \"id\": string,\n"
    "      \"question\": string,\n"
    "      \"options\": [string, string, string, string],\n"
    "      \"answer_index\": integer,\n"
    "      \"explanation\": string\n"
    "    }\n"
    "  ]\n"
    "}\n"
)

def render_quiz_form(quiz: Dict[str, Any]) -> Tuple[Dict[str, int], int, int]:
    answers: Dict[str, int] = {}
    score = 0
    total = 0
    for q in quiz.get("questions", []):
        qid = q.get("id")
        st.markdown(f"**{q.get('question','')}**")
        choice = st.radio(
            label="Select one:",
            options=list(range(len(q.get("options", [])))),
            format_func=lambda i, opts=q.get("options", []): (opts[i] if i < len(opts) else str(i)),
            key=f"ans_{qid}",
            horizontal=False,
        )
        answers[qid] = int(choice)
        total += 1
        if int(choice) == int(q.get("answer_index", -1)):
            score += 1
    return answers, score, total

# ---- Streamlit UI ----
st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy")
st.markdown("Upload your syllabus/book PDF and ask questions. Now with SQL storage for Q&A, quizzes, and results.")

with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="guest")
    llm_choice = st.radio("LLM:", ["Gemini", "OpenAI"], index=0)
    language = st.selectbox("Answer language:", ["Auto (detect)", "English", "Hindi"])
    use_voice_in = st.checkbox("Enable voice input", value=False)
    use_voice_out = st.checkbox("Enable voice output", value=False)
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload syllabus PDF", type=["pdf"])

# --- Handle upload/indexing ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    st.sidebar.success("PDF uploaded")
    text = extract_text_from_pdf(tmp_path)
    with st.spinner("Indexing PDF and storing chunks+embeddings..."):
        vectordb, docs, _ = create_vector_store_from_text(text, index_path=INDEX_PATH)
        embedder = get_embedder()
        embeddings = embedder.embed_documents(docs)
        doc_id = os.path.basename(uploaded_file.name)
        sql_saver.save_document_chunks(user_id=user_id, doc_id=doc_id, chunks=docs, embeddings=embeddings)
    st.sidebar.success("Indexed and stored!")

# ---- Load/Build QA chain ----
if os.path.exists(INDEX_PATH):
    vectordb = load_vector_store(INDEX_PATH)
    qa_chain = build_qa_chain(vectordb, llm_choice=llm_choice)
else:
    vectordb = None
    qa_chain = None

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Chat / Ask questions")
    if qa_chain:
        user_query = ""
        if use_voice_in and st.button("Record voice query"):
            lang_code = "hi-IN" if language.startswith("Hindi") else "en-IN"
            res = recognize_speech_from_mic(lang_code=lang_code)
            if res:
                user_query = res
                st.write("You said:", res)
        typed = st.text_input("Or type your question here:")
        if typed:
            user_query = typed
        if st.button("Get Answer") and user_query:
            with st.spinner("Retrieving answer..."):
                answer = qa_chain.run(user_query)
            answer_out = translate_text(answer, "hi") if language == "Hindi" else answer
            st.markdown("**Answer:**")
            st.write(answer_out)
            sql_saver.save({
                "type": "qa",
                "question": user_query,
                "answer": answer_out,
                "llm_choice": llm_choice,
                "timestamp": time.time(),
            }, user_id=user_id)
            if use_voice_out:
                audio_file = speak_text(answer_out, lang="hi" if language.startswith("Hindi") else "en")
                if audio_file:
                    st.audio(audio_file, format="audio/mp3")
    else:
        st.info("Please upload a PDF first.")

with col2:
    st.subheader("Quiz Generator")
    if vectordb:
        topic = st.text_input("Topic / chapter:")
        num_q = st.slider("Number of questions", 1, 10, 5)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        if st.button("Generate Quiz"):
            docs = vectordb.similarity_search(topic or "Summarize main points", k=6)
            context_text = "\n".join([d.page_content for d in docs])
            quiz_prompt = f"""
You are an expert teacher. Based on the CONTEXT below, generate exactly {num_q} MCQs (4 options each).
Each question must include: id, question, options[4], answer_index, and explanation.
{QUIZ_JSON_INSTRUCTIONS}
CONTEXT:\n{context_text}
"""
            if llm_choice == "OpenAI":
                llm = ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
                quiz_response = llm.predict(quiz_prompt)
            else:
                llm = ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0)
                quiz_response = llm.predict(quiz_prompt)
            quiz_json = safe_json_loads(quiz_response)
            st.success("Quiz generated! Scroll down to attempt it.")
            sql_saver.save({"type": "quiz", "topic": topic, "difficulty": difficulty, "quiz": quiz_json, "timestamp": time.time()}, user_id=user_id)
            quiz_id = sql_saver.save_quiz(user_id=user_id, topic=topic, difficulty=difficulty, quiz_json=quiz_json)
