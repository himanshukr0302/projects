# build_index.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "models/embedding-001"
INDEX_PATH = "faiss_index"

pdf_path = "my_syllabus.pdf"  # <-- Change to your PDF file

# 1. Load the PDF
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# 4. Build FAISS vector store
vectordb = FAISS.from_documents(chunks, embeddings)

# 5. Save index
vectordb.save_local(INDEX_PATH)
print(f"✅ Index saved to {INDEX_PATH}")
