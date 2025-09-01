import re
import json
from typing import Dict, Any

def safe_json_loads(data):
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return {}
    try:
        cleaned = re.sub(r"```(?:json)?", "", data, flags=re.IGNORECASE).strip("` \n\r\t")
        return json.loads(cleaned)
    except Exception:
        return {}

def make_quiz_prompt(context_text: str, num_q: int, instructions: str) -> str:
    return (
        f"You are an expert teacher. Based on the CONTEXT below, generate exactly {num_q} MCQs (4 options each).\n"
        f"Each question must include: id, question, options[4], answer_index, and explanation.\n"
        f"{instructions}\n"
        f"CONTEXT:\n{context_text}\n"
    )

def generate_quiz_json(llm, vectordb, topic: str, num_q: int, instructions: str) -> Dict[str, Any]:
    docs = vectordb.similarity_search(topic or "Main points", k=6)
    context_text = "\n".join([d.page_content for d in docs])
    prompt = make_quiz_prompt(context_text, num_q, instructions)
    raw = llm.predict(prompt)
    return safe_json_loads(raw)
