import os
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import HF_EMBED_MODEL, INDEX_PATH

def extract_text_from_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def create_index_from_text(text: str, chunk_size=1500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_text(text)
    embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    vectordb = FAISS.from_texts(docs, embedding=embedder)
    vectordb.save_local(INDEX_PATH)
    return vectordb, docs, embedder

def load_index():
    embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
    return FAISS.load_local(INDEX_PATH, embedder, allow_dangerous_deserialization=True)
