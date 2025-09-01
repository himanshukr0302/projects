import json
import pickle
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple
from config import DB_PATH

def _safe_json_loads(data):
    if isinstance(data, dict):
        return data
    if not isinstance(data, str):
        return {}
    try:
        import re
        cleaned = re.sub(r"```(?:json)?", "", data, flags=re.IGNORECASE).strip("` \n\r\t")
        return json.loads(cleaned)
    except Exception:
        return {}

class SQLSaver:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS items (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          type TEXT NOT NULL,
          user_id TEXT DEFAULT 'guest',
          payload TEXT NOT NULL,
          created_at REAL NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT DEFAULT 'guest',
          doc_id TEXT,
          chunk_id INTEGER,
          content TEXT,
          embedding BLOB
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS quizzes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT DEFAULT 'guest',
          topic TEXT,
          difficulty TEXT,
          quiz_json TEXT NOT NULL,
          created_at REAL NOT NULL
        )""")
        c.execute("""
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

    # generic items
    def save(self, obj: Dict[str, Any], user_id: str = "guest"):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO items(type, user_id, payload, created_at) VALUES (?, ?, ?, ?)",
            (obj.get("type", "unknown"), user_id, json.dumps(obj), time.time()),
        )
        self.conn.commit()

    # documents
    def save_document_chunks(self, user_id: str, doc_id: str, chunks: List[str], embeddings: List[List[float]]):
        c = self.conn.cursor()
        for idx, (txt, emb) in enumerate(zip(chunks, embeddings)):
            c.execute(
                "INSERT INTO documents(user_id, doc_id, chunk_id, content, embedding) VALUES (?, ?, ?, ?, ?)",
                (user_id, doc_id, idx, txt, pickle.dumps(emb)),
            )
        self.conn.commit()

    # quizzes
    def save_quiz(self, user_id: str, topic: str, difficulty: str, quiz_json: Dict[str, Any]) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO quizzes(user_id, topic, difficulty, quiz_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, topic, difficulty, json.dumps(quiz_json), time.time()),
        )
        qid = c.lastrowid
        self.conn.commit()
        return qid

    def list_quizzes(self, user_id: str, limit: int = 25) -> List[Tuple[int, str, str, float]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, topic, difficulty, created_at FROM quizzes WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return c.fetchall()

    def get_quiz(self, quiz_id: int) -> Optional[Dict[str, Any]]:
        c = self.conn.cursor()
        c.execute("SELECT quiz_json FROM quizzes WHERE id=?", (quiz_id,))
        r = c.fetchone()
        return _safe_json_loads(r[0]) if r else None

    # results
    def save_quiz_result(self, user_id: str, quiz_id: int, answers: Dict[str, Any], score: int, total: int) -> int:
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO quiz_results(user_id, quiz_id, answers_json, score, total, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, quiz_id, json.dumps(answers), score, total, time.time()),
        )
        rid = c.lastrowid
        self.conn.commit()
        return rid

    def list_results(self, user_id: str, limit: int = 25) -> List[Tuple[int, int, int, int, float]]:
        c = self.conn.cursor()
        c.execute(
            "SELECT id, quiz_id, score, total, created_at FROM quiz_results WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return c.fetchall()

# one shared connection
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
sql_saver = SQLSaver(conn)
