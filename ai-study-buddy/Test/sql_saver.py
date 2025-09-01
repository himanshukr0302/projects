# sql_saver.py
import sqlite3

class SQLSaver:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS study_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT,
                question TEXT,
                answer TEXT,
                llm_choice TEXT,
                topic TEXT,
                difficulty TEXT,
                quiz TEXT,
                timestamp REAL
            )
        """)
        self.conn.commit()

    def save(self, data: dict):
        self.conn.execute("""
            INSERT INTO study_data (type, question, answer, llm_choice, topic, difficulty, quiz, timestamp)
            VALUES (:type, :question, :answer, :llm_choice, :topic, :difficulty, :quiz, :timestamp)
        """, {
            "type": data.get("type"),
            "question": data.get("question"),
            "answer": data.get("answer"),
            "llm_choice": data.get("llm_choice"),
            "topic": data.get("topic"),
            "difficulty": data.get("difficulty"),
            "quiz": data.get("quiz"),
            "timestamp": data.get("timestamp"),
        })
        self.conn.commit()

    def load_all(self):
        cur = self.conn.execute("SELECT * FROM study_data ORDER BY timestamp DESC")
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
