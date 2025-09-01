import os
import re
import json
import pypdf
import tempfile
from typing import Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import HF_EMBED_MODEL, INDEX_PATH

def safe_json_loads(data):
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return {}
    try:
        cleaned = re.sub(r"```(?:json)?", "", data, flags=re.IGNORECASE).strip("` \n\r\t")
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

def extract_text_from_pdf(file_path: str) -> str:
    reader = pypdf.PdfReader(file_path)
    return "\n".join([p.extract_text() or "" for p in reader.pages])

def create_vector_store_from_text(text: str, chunk_size=1500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectordb = FAISS.from_texts(docs, embedding=embeddings)
    vectordb.save_local(INDEX_PATH)
    return vectordb, docs, embeddings

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
