from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class ChatTurn:
    id: int
    user_id: str
    session_id: str
    episode_id: Optional[str]
    query: str
    answer: str
    policy_notes: list[str]
    references: list[str]
    body_tags: list[str]
    intent_tags: list[str]
    created_at: str


@dataclass(frozen=True)
class EpisodeState:
    episode_id: str
    user_id: str
    session_id: str
    status: str
    summary: str
    slots: dict[str, Any]
    created_at: str
    updated_at: str
    closed_at: Optional[str]


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
        episode_id: Optional[str] = None,
    ) -> int:
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

    def get_active_episode(self, *, session_id: str) -> Optional[EpisodeState]:
        ...

    def start_episode(self, *, user_id: str, session_id: str, summary: str = "") -> EpisodeState:
        ...

    def close_active_episode(self, *, session_id: str) -> None:
        ...

    def update_episode_slots(
        self,
        *,
        episode_id: str,
        updates: dict[str, Any],
        source_turn_id: Optional[int] = None,
    ) -> EpisodeState:
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
        episode_id: Optional[str] = None,
    ) -> int:
        return 0

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

    def get_active_episode(self, *, session_id: str) -> Optional[EpisodeState]:
        return None

    def start_episode(self, *, user_id: str, session_id: str, summary: str = "") -> EpisodeState:
        return EpisodeState(
            episode_id="ep_null",
            user_id=user_id,
            session_id=session_id,
            status="active",
            summary=summary,
            slots={},
            created_at="",
            updated_at="",
            closed_at=None,
        )

    def close_active_episode(self, *, session_id: str) -> None:
        return

    def update_episode_slots(
        self,
        *,
        episode_id: str,
        updates: dict[str, Any],
        source_turn_id: Optional[int] = None,
    ) -> EpisodeState:
        return EpisodeState(
            episode_id=episode_id or "ep_null",
            user_id="",
            session_id="",
            status="active",
            summary="",
            slots=updates,
            created_at="",
            updated_at="",
            closed_at=None,
        )
