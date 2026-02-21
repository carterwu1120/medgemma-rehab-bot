from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class ChatTurn:
    user_id: str
    session_id: str
    query: str
    answer: str
    policy_notes: list[str]
    references: list[str]
    body_tags: list[str]
    intent_tags: list[str]
    created_at: str


class ChatMemoryStore(Protocol):
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
        ...

    def get_recent_session_turns(self, *, session_id: str, limit: int = 6) -> list[ChatTurn]:
        ...

    def get_recent_user_turns(
        self,
        *,
        user_id: str,
        limit: int = 6,
        exclude_session_id: Optional[str] = None,
    ) -> list[ChatTurn]:
        ...


class NullMemoryStore:
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
        return

    def get_recent_session_turns(self, *, session_id: str, limit: int = 6) -> list[ChatTurn]:
        return []

    def get_recent_user_turns(
        self,
        *,
        user_id: str,
        limit: int = 6,
        exclude_session_id: Optional[str] = None,
    ) -> list[ChatTurn]:
        return []
