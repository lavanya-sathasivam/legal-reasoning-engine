import json
import sqlite3
from pathlib import Path
from typing import Any

from src.config import APP_DB_PATH


class PlatformStore:
    def __init__(self, db_path: Path = APP_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS firms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL DEFAULT 'Default Firm',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    firm_id INTEGER NOT NULL DEFAULT 1,
                    name TEXT NOT NULL DEFAULT 'Demo Lawyer',
                    role TEXT NOT NULL DEFAULT 'lawyer',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS matters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    firm_id INTEGER NOT NULL DEFAULT 1,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matter_id INTEGER,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    analysis_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matter_id INTEGER,
                    input_text TEXT NOT NULL,
                    extracted_facts_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    matter_id INTEGER,
                    source_id TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL
                );
                """
            )
            connection.execute("INSERT OR IGNORE INTO firms (id, name) VALUES (1, 'Default Firm')")
            connection.execute("INSERT OR IGNORE INTO users (id, firm_id, name, role) VALUES (1, 1, 'Demo Lawyer', 'lawyer')")
            defaults = {
                "ai_provider": "deterministic",
                "enabled_laws": ["BNS", "BNSS", "BSA", "Constitution"],
                "confidence_threshold": 0.43,
                "theme": "light",
                "citation_style": "law-section",
            }
            for key, value in defaults.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings (key, value_json) VALUES (?, ?)",
                    (key, json.dumps(value)),
                )

    def list_matters(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM matters ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def create_matter(self, title: str, description: str = "") -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO matters (title, description) VALUES (?, ?)",
                (title, description),
            )
            row = connection.execute("SELECT * FROM matters WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def save_message(self, role: str, content: str, matter_id: int | None = None, analysis: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO messages (matter_id, role, content, analysis_json) VALUES (?, ?, ?, ?)",
                (matter_id, role, content, json.dumps(analysis) if analysis else None),
            )
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def save_analysis(self, input_text: str, extracted_facts: dict[str, Any], recommendations: list[dict[str, Any]], matter_id: int | None = None) -> dict[str, Any]:
        confidence = max((item.get("confidence", 0.0) for item in recommendations), default=0.0)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO analysis_runs (matter_id, input_text, extracted_facts_json, recommendations_json, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (matter_id, input_text, json.dumps(extracted_facts), json.dumps(recommendations), confidence),
            )
            row = connection.execute("SELECT * FROM analysis_runs WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def get_settings(self) -> dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value_json FROM settings").fetchall()
        return {row["key"]: json.loads(row["value_json"]) for row in rows}

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as connection:
            for key, value in updates.items():
                connection.execute(
                    "INSERT INTO settings (key, value_json) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
                    (key, json.dumps(value)),
                )
        return self.get_settings()
