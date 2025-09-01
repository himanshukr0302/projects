import sqlite3
from datetime import datetime
import json

DB_PATH = "studybuddy.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Store uploaded documents
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            filename TEXT,
            upload_date TEXT
        )
    ''')
    
    # Store Q&A per document
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER,
            question TEXT,
            answer TEXT,
            timestamp TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_document(user_id, filename):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO documents (user_id, filename, upload_date) VALUES (?, ?, ?)',
              (user_id, filename, datetime.now().isoformat()))
    doc_id = c.lastrowid
    conn.commit()
    conn.close()
    return doc_id

def save_qa(doc_id, question, answer):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('INSERT INTO document_qa (doc_id, question, answer, timestamp) VALUES (?, ?, ?, ?)',
              (doc_id, question, answer, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_document_history(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, filename, upload_date FROM documents WHERE user_id = ?', (user_id,))
    docs = c.fetchall()
    conn.close()
    return docs

def get_doc_qa(doc_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT question, answer, timestamp FROM document_qa WHERE doc_id = ?', (doc_id,))
    qa_list = c.fetchall()
    conn.close()
    return qa_list
