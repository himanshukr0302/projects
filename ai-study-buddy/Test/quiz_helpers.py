import streamlit as st
from typing import Dict, Any, Tuple

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
