import os
import tempfile
import time
import streamlit as st
from config import QUIZ_JSON_INSTRUCTIONS, INDEX_PATH
from data_storage import sql_saver
from pdf_handler import extract_text_from_pdf, create_index_from_text, load_index
from qa_engine import build_qa_chain, build_llm
from quiz_generator import generate_quiz_json

# --- Streamlit base ---
st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy")
st.caption("Upload a PDF, ask questions, generate quizzes, and keep your history.")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    user_id = st.text_input("User ID", value="guest")
    llm_choice = st.radio("LLM:", ["Gemini", "OpenAI"], index=0)
    uploaded = st.file_uploader("Upload syllabus PDF", type=["pdf"])
    st.divider()

# --- Session defaults ---
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "current_quiz_id" not in st.session_state:
    st.session_state.current_quiz_id = None

# --- Upload & index ---
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    text = extract_text_from_pdf(path)
    with st.spinner("Indexing PDF & storing chunks..."):
        vectordb, docs, embedder = create_index_from_text(text)
        embeddings = embedder.embed_documents(docs)
        sql_saver.save_document_chunks(user_id, os.path.basename(uploaded.name), docs, embeddings)
    st.sidebar.success("Indexed and stored!")

# --- Load index & QA chain if exists ---
vectordb = qa_chain = llm = None
if os.path.exists(INDEX_PATH):
    vectordb = load_index()
    qa_chain, llm = build_qa_chain(vectordb, llm_choice)

col1, col2 = st.columns([2, 1])

# =========================
# Q&A (left)
# =========================
with col1:
    st.subheader("Chat / Ask questions")
    if qa_chain:
        q = st.text_input("Ask a question about your PDF:")
        if st.button("Get Answer"):
            if not q.strip():
                st.warning("Please type a question.")
            else:
                with st.spinner("Retrieving..."):
                    ans = qa_chain.run(q)
                st.markdown("**Answer:**")
                st.write(ans)
                sql_saver.save(
                    {"type": "qa", "question": q, "answer": ans, "llm_choice": llm_choice, "timestamp": time.time()},
                    user_id=user_id,
                )
    else:
        st.info("Upload a PDF to enable Q&A.")

# =========================
# Quiz (right)
# =========================
with col2:
    st.subheader("Quiz Generator")
    if vectordb and llm:
        topic = st.text_input("Topic / chapter:")
        num_q = st.slider("Number of questions", 1, 10, 5)
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        if st.button("Generate Quiz"):
            with st.spinner("Generating quiz..."):
                quiz_json = generate_quiz_json(llm, vectordb, topic, num_q, QUIZ_JSON_INSTRUCTIONS)
            if not quiz_json or "questions" not in quiz_json or not quiz_json["questions"]:
                st.error("Quiz generation failed. Try a different topic or model.")
            else:
                qid = sql_saver.save_quiz(user_id, topic or "General", difficulty, quiz_json)
                st.session_state.current_quiz = quiz_json
                st.session_state.current_quiz_id = qid
                st.success("Quiz generated! Scroll down to attempt it.")
    else:
        st.info("Index a PDF first to generate quizzes.")

# =========================
# Attempt current quiz (global section)
# =========================
st.markdown("### Attempt Quiz")
if st.session_state.current_quiz:
    answers = {}
    score = 0
    total = 0
    for q in st.session_state.current_quiz.get("questions", []):
        qid = q.get("id") or f"q{total+1}"
        st.markdown(f"**{q.get('question','')}**")
        choice = st.radio(
            "Select one:",
            options=list(range(len(q.get("options", [])))),
            format_func=lambda i, opts=q.get("options", []): opts[i] if i < len(opts) else str(i),
            key=f"ans_{qid}",
        )
        answers[qid] = int(choice)
        total += 1
        if int(choice) == int(q.get("answer_index", -1)):
            score += 1

    if st.button("Submit Quiz"):
        sql_saver.save_quiz_result(user_id, st.session_state.current_quiz_id, answers, score, total)
        st.success(f"Score: {score}/{total}")
else:
    st.info("Generate a quiz or load one from history to attempt it.")

st.divider()

# =========================
# History panels
# =========================
c1, c2 = st.columns(2)

with c1:
    st.subheader("📜 Quiz History")
    if user_id.strip():
        rows = sql_saver.list_quizzes(user_id, limit=20)
        if not rows:
            st.write("No quizzes yet.")
        else:
            for qid, topic, diff, ts in rows:
                if st.button(f"Load: {topic} • {diff}", key=f"load_{qid}"):
                    qj = sql_saver.get_quiz(qid)
                    if qj:
                        st.session_state.current_quiz = qj
                        st.session_state.current_quiz_id = qid
                        st.success(f"Loaded quiz #{qid}")
    else:
        st.info("Enter a user id to view history.")

with c2:
    st.subheader("🏁 Recent Results")
    rows = sql_saver.list_results(user_id, limit=20)
    if not rows:
        st.write("No quiz results yet.")
    else:
        for rid, qid, score, total, ts in rows:
            st.write(f"Result #{rid} • Quiz {qid} → **{score}/{total}**")
