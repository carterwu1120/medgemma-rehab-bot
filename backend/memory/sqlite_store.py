from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .store import ChatTurn


class SQLiteChatMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    policy_notes TEXT NOT NULL DEFAULT '[]',
                    references_json TEXT NOT NULL DEFAULT '[]',
                    body_tags_json TEXT NOT NULL DEFAULT '[]',
                    intent_tags_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_chat_turns_session_created
                    ON chat_turns (session_id, created_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_chat_turns_user_created
                    ON chat_turns (user_id, created_at DESC, id DESC);
                """
            )

    @staticmethod
    def _to_json(value: list[str]) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _from_json(value: str) -> list[str]:
        try:
            data = json.loads(value)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> ChatTurn:
        return ChatTurn(
            user_id=row["user_id"],
            session_id=row["session_id"],
            query=row["query"],
            answer=row["answer"],
            policy_notes=SQLiteChatMemoryStore._from_json(row["policy_notes"]),
            references=SQLiteChatMemoryStore._from_json(row["references_json"]),
            body_tags=SQLiteChatMemoryStore._from_json(row["body_tags_json"]),
            intent_tags=SQLiteChatMemoryStore._from_json(row["intent_tags_json"]),
            created_at=row["created_at"],
        )

    def add_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        policy_notes: list[str],
        references: list[str],
        body_tags: list[str],
        intent_tags: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_turns (
                    user_id,
                    session_id,
                    query,
                    answer,
                    policy_notes,
                    references_json,
                    body_tags_json,
                    intent_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    query,
                    answer,
                    self._to_json(policy_notes),
                    self._to_json(references),
                    self._to_json(body_tags),
                    self._to_json(intent_tags),
                ),
            )

    def get_recent_session_turns(self, *, session_id: str, limit: int = 6) -> list[ChatTurn]:
        capped_limit = max(1, min(limit, 20))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT user_id, session_id, query, answer, policy_notes, references_json,
                       body_tags_json, intent_tags_json, created_at
                FROM chat_turns
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, capped_limit),
            ).fetchall()
        turns = [self._row_to_turn(row) for row in rows]
        turns.reverse()
        return turns

    def get_recent_user_turns(
        self,
        *,
        user_id: str,
        limit: int = 6,
        exclude_session_id: Optional[str] = None,
    ) -> list[ChatTurn]:
        capped_limit = max(1, min(limit, 20))
        with self._connect() as conn:
            if exclude_session_id:
                rows = conn.execute(
                    """
                    SELECT user_id, session_id, query, answer, policy_notes, references_json,
                           body_tags_json, intent_tags_json, created_at
                    FROM chat_turns
                    WHERE user_id = ? AND session_id != ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, exclude_session_id, capped_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT user_id, session_id, query, answer, policy_notes, references_json,
                           body_tags_json, intent_tags_json, created_at
                    FROM chat_turns
                    WHERE user_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (user_id, capped_limit),
                ).fetchall()
        turns = [self._row_to_turn(row) for row in rows]
        turns.reverse()
        return turns
