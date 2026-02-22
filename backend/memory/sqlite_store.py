from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .store import ChatTurn, EpisodeState


class SQLiteChatMemoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_json(value: list[str] | dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def _from_json(value: str) -> list[Any] | dict[str, Any]:
        try:
            data = json.loads(value)
            if isinstance(data, (list, dict)):
                return data
            return []
        except json.JSONDecodeError:
            return []

    def _ensure_chat_turns_episode_column(self, conn: sqlite3.Connection) -> None:
        cols = conn.execute("PRAGMA table_info(chat_turns)").fetchall()
        names = {c["name"] for c in cols}
        if "episode_id" not in names:
            conn.execute("ALTER TABLE chat_turns ADD COLUMN episode_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_turns_episode_created "
                "ON chat_turns (episode_id, created_at DESC, id DESC)"
            )

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

                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL UNIQUE,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    summary TEXT NOT NULL DEFAULT '',
                    slots_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                    closed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_episodes_session_status_updated
                    ON episodes (session_id, status, updated_at DESC, id DESC);

                CREATE INDEX IF NOT EXISTS idx_episodes_user_updated
                    ON episodes (user_id, updated_at DESC, id DESC);

                CREATE TABLE IF NOT EXISTS episode_slot_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    slot_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    source_turn_id INTEGER,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_episode_slot_history_episode_created
                    ON episode_slot_history (episode_id, created_at DESC, id DESC);
                """
            )
            self._ensure_chat_turns_episode_column(conn)

    @staticmethod
    def _row_to_turn(row: sqlite3.Row) -> ChatTurn:
        policy_notes = SQLiteChatMemoryStore._from_json(row["policy_notes"])
        references = SQLiteChatMemoryStore._from_json(row["references_json"])
        body_tags = SQLiteChatMemoryStore._from_json(row["body_tags_json"])
        intent_tags = SQLiteChatMemoryStore._from_json(row["intent_tags_json"])
        return ChatTurn(
            id=int(row["id"]),
            user_id=row["user_id"],
            session_id=row["session_id"],
            episode_id=row["episode_id"],
            query=row["query"],
            answer=row["answer"],
            policy_notes=policy_notes if isinstance(policy_notes, list) else [],
            references=references if isinstance(references, list) else [],
            body_tags=body_tags if isinstance(body_tags, list) else [],
            intent_tags=intent_tags if isinstance(intent_tags, list) else [],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_episode(row: sqlite3.Row) -> EpisodeState:
        slots = SQLiteChatMemoryStore._from_json(row["slots_json"])
        return EpisodeState(
            episode_id=row["episode_id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            status=row["status"],
            summary=row["summary"],
            slots=slots if isinstance(slots, dict) else {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            closed_at=row["closed_at"],
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
        episode_id: Optional[str] = None,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_turns (
                    user_id,
                    session_id,
                    episode_id,
                    query,
                    answer,
                    policy_notes,
                    references_json,
                    body_tags_json,
                    intent_tags_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    session_id,
                    episode_id,
                    query,
                    answer,
                    self._to_json(policy_notes),
                    self._to_json(references),
                    self._to_json(body_tags),
                    self._to_json(intent_tags),
                ),
            )
            return int(cur.lastrowid)

    def get_recent_session_turns(self, *, session_id: str, limit: int = 6) -> list[ChatTurn]:
        capped_limit = max(1, min(limit, 20))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, user_id, session_id, episode_id, query, answer, policy_notes, references_json,
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
                    SELECT id, user_id, session_id, episode_id, query, answer, policy_notes, references_json,
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
                    SELECT id, user_id, session_id, episode_id, query, answer, policy_notes, references_json,
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

    def get_active_episode(self, *, session_id: str) -> Optional[EpisodeState]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT episode_id, user_id, session_id, status, summary, slots_json, created_at, updated_at, closed_at
                FROM episodes
                WHERE session_id = ? AND status = 'active'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_episode(row)

    def start_episode(self, *, user_id: str, session_id: str, summary: str = "") -> EpisodeState:
        episode_id = f"ep_{uuid4().hex}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodes (
                    episode_id,
                    user_id,
                    session_id,
                    status,
                    summary,
                    slots_json
                ) VALUES (?, ?, ?, 'active', ?, '{}')
                """,
                (episode_id, user_id, session_id, summary.strip()),
            )
            row = conn.execute(
                """
                SELECT episode_id, user_id, session_id, status, summary, slots_json, created_at, updated_at, closed_at
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
        return self._row_to_episode(row)

    def close_active_episode(self, *, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET status = 'closed',
                    closed_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE session_id = ? AND status = 'active'
                """,
                (session_id,),
            )

    def update_episode_slots(
        self,
        *,
        episode_id: str,
        updates: dict[str, Any],
        source_turn_id: Optional[int] = None,
    ) -> EpisodeState:
        if not updates:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT episode_id, user_id, session_id, status, summary, slots_json, created_at, updated_at, closed_at
                    FROM episodes
                    WHERE episode_id = ?
                    """,
                    (episode_id,),
                ).fetchone()
            return self._row_to_episode(row)

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT episode_id, user_id, session_id, status, summary, slots_json, created_at, updated_at, closed_at
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()
            if not row:
                raise RuntimeError(f"episode not found: {episode_id}")
            current = self._row_to_episode(row)
            merged_slots = dict(current.slots)

            for key, new_val in updates.items():
                old_val = merged_slots.get(key)
                if old_val == new_val:
                    continue
                merged_slots[key] = new_val
                conn.execute(
                    """
                    INSERT INTO episode_slot_history (
                        episode_id,
                        slot_name,
                        old_value,
                        new_value,
                        source_turn_id
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        episode_id,
                        key,
                        "" if old_val is None else str(old_val),
                        "" if new_val is None else str(new_val),
                        source_turn_id,
                    ),
                )

            conn.execute(
                """
                UPDATE episodes
                SET slots_json = ?,
                    updated_at = datetime('now')
                WHERE episode_id = ?
                """,
                (self._to_json(merged_slots), episode_id),
            )

            updated_row = conn.execute(
                """
                SELECT episode_id, user_id, session_id, status, summary, slots_json, created_at, updated_at, closed_at
                FROM episodes
                WHERE episode_id = ?
                """,
                (episode_id,),
            ).fetchone()

        return self._row_to_episode(updated_row)
