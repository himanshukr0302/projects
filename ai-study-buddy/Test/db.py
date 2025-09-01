import json
import pickle
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
from utils import safe_json_loads

class SQLSaver:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            user_id TEXT DEFAULT 'guest',
            payload TEXT NOT NULL,
            created_at REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'guest',
            doc_id TEXT,
            chunk_id INTEGER,
            content TEXT,
            embedding BLOB
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'guest',
            topic TEXT,
            difficulty TEXT,
            quiz_json TEXT NOT NULL,
            created_at REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT DEFAULT 'guest',
            quiz_id INTEGER,
            answers_json TEXT,
            score INTEGER,
            total INTEGER,
            created_at REAL NOT NULL,
            FOREIGN KEY(quiz_id) REFERENCES quizzes(id)
        )""")
        self.conn.commit()

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

    def save_document_chunks(self, user_id: str, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
        cur = self.conn.cursor()
        for idx, (text, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO documents(user_id, doc_id, chunk_id, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (user_id, doc_id, idx, text, pickle.dumps(emb)),
            )
        self.conn.commit()

    def save_quiz(self, user_id: str, topic: str, difficulty: str, quiz_json: Dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO quizzes(user_id, topic, difficulty, quiz_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, topic, difficulty, json.dumps(quiz_json), time.time()),
        )
        quiz_id = cur.lastrowid
        self.conn.commit()
        return quiz_id

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

# Initialize DB connection
conn = sqlite3.connect("studybuddy.db", check_same_thread=False)
sql_saver = SQLSaver(conn)
