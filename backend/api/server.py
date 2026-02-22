import os
import re
from functools import lru_cache
from typing import List, Optional, Set
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.llm_api import create_vllm_api
from backend.memory import (
    ChatMemoryStore,
    ChatTurn,
    NullMemoryStore,
    SQLiteChatMemoryStore,
    build_episode_context,
    detect_body_bucket,
    extract_slot_updates,
    has_minimum_slots_for_plan,
    resolve_follow_up_query,
    should_start_new_episode,
)
from backend.rag import BM25Retriever, RehabRAG, RetrievedChunk
from backend.recommendation import (
    CatalogVideoProvider,
    VideoCandidate,
    VideoRecommender,
    YouTubeDataApiProvider,
    YouTubeSearchProvider,
    detect_query_language,
    infer_body_tags,
    infer_intent_tags,
    summarize_video,
)


CHUNK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+_p\d+_c\d+")
REQUEST_LIKE_PATTERN = re.compile(
    r"(請|可以|怎麼|如何|給我|建議|計畫|方案|should|what|how|can you|advice|plan|\?)",
    re.IGNORECASE,
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    video_limit: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    max_tokens: int = Field(default=450, ge=64, le=2048)
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    session_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    source_name: str
    page: int
    score: float
    tags: List[str]
    snippet: str


class VideoOut(BaseModel):
    video_id: str
    title: str
    url: str
    provider: str
    score: float
    summary: str
    tags: List[str]
    intent_tags: List[str]
    why: List[str]
    notes: str


class ChatResponse(BaseModel):
    user_id: str
    session_id: str
    episode_id: Optional[str] = None
    history_turns_used: int
    effective_query: str
    answer: str
    language: str
    policy_notes: List[str]
    references: List[str]
    episode_slots: dict[str, str]
    body_tags: List[str]
    intent_tags: List[str]
    retrieved_chunks: List[RetrievedChunkOut]
    videos: List[VideoOut]


def _extract_chunk_refs(text: str) -> List[str]:
    return sorted(set(CHUNK_ID_PATTERN.findall(text)))


def _clip(text: str, limit: int = 220) -> str:
    compact = " ".join(text.strip().split())
    if len(compact) <= limit:
        return compact
    return compact[: max(1, limit - 1)] + "…"


def _looks_like_clarification_answer(answer: str, notes: List[str]) -> bool:
    lowered = answer.lower()
    if "clarification_mode" in notes or "clarify_first_only" in notes:
        return True
    return (
        "我先確認一件事" in answer
        or "先確認" in answer
        or "one quick check" in lowered
        or "please confirm" in lowered
    )


def _looks_like_short_followup(query: str) -> bool:
    compact = " ".join(query.strip().split())
    if not compact:
        return False
    if len(compact) > 64:
        return False
    if REQUEST_LIKE_PATTERN.search(compact):
        return False
    return True


def _merge_recent_user_queries(session_turns: List[ChatTurn], latest_query: str, max_queries: int = 3) -> str:
    recent_user_queries = [turn.query.strip() for turn in session_turns[-max_queries:] if turn.query.strip()]
    merged_parts: List[str] = []
    seen: Set[str] = set()
    for text in recent_user_queries:
        norm = " ".join(text.split())
        if norm and norm not in seen:
            seen.add(norm)
            merged_parts.append(norm)
    latest_norm = " ".join(latest_query.strip().split())
    if latest_norm and latest_norm not in seen:
        merged_parts.append(latest_norm)
    return "\n".join(merged_parts)


def _build_history_context(
    *,
    session_turns: List[ChatTurn],
    user_turns: List[ChatTurn],
    max_chars: int = 1800,
) -> Optional[str]:
    if not session_turns and not user_turns:
        return None

    lines: List[str] = []

    include_assistant = os.getenv("CHAT_HISTORY_INCLUDE_ASSISTANT", "0") == "1"

    if session_turns:
        lines.append("Current session history:")
        for idx, turn in enumerate(session_turns[-4:], start=1):
            lines.append(f"- S{idx} User: {_clip(turn.query, 200)}")
            if include_assistant:
                lines.append(f"  S{idx} Assistant: {_clip(turn.answer, 260)}")

    if user_turns:
        lines.append("User long-term history from previous sessions:")
        for idx, turn in enumerate(user_turns[-4:], start=1):
            lines.append(
                f"- U{idx} ({turn.session_id}) User: {_clip(turn.query, 160)} | "
                f"Assistant: {_clip(turn.answer, 200)}"
            )

    context = "\n".join(lines).strip()
    if len(context) > max_chars:
        context = context[: max_chars - 1] + "…"
    return context


def _filter_turns_by_episode(turns: List[ChatTurn], episode_id: Optional[str]) -> List[ChatTurn]:
    if not episode_id:
        return []
    return [turn for turn in turns if turn.episode_id == episode_id]


def _chunk_to_out(chunk: RetrievedChunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        source_name=chunk.source_name,
        page=chunk.page,
        score=round(float(chunk.score), 6),
        tags=list(chunk.tags),
        snippet=chunk.text[:320],
    )


def _video_to_out(video: VideoCandidate, language: str) -> VideoOut:
    return VideoOut(
        video_id=video.video_id,
        title=video.title,
        url=video.url,
        provider=video.provider,
        score=round(float(video.score), 6),
        summary=summarize_video(video, language=language),
        tags=list(video.tags),
        intent_tags=list(video.intent_tags),
        why=list(video.why),
        notes=video.notes,
    )


def _build_video_recommender() -> VideoRecommender:
    providers = [CatalogVideoProvider(os.getenv("VIDEO_CATALOG", "configs/video_catalog.sample.jsonl"))]

    use_youtube_api = os.getenv("VIDEO_USE_YOUTUBE_API", "1") == "1"
    channel_whitelist_raw = os.getenv("YOUTUBE_CHANNEL_WHITELIST", "")
    channel_whitelist = [x.strip() for x in channel_whitelist_raw.split(",") if x.strip()]
    youtube_api = YouTubeDataApiProvider(
        api_key=os.getenv("YOUTUBE_API_KEY", ""),
        region_code=os.getenv("YOUTUBE_REGION", "TW"),
        channel_whitelist=channel_whitelist,
    )

    if use_youtube_api and youtube_api.enabled:
        providers.append(youtube_api)
    else:
        providers.append(YouTubeSearchProvider())
    return VideoRecommender(providers)


def _build_memory_store() -> ChatMemoryStore:
    if os.getenv("CHAT_MEMORY_ENABLED", "1") != "1":
        return NullMemoryStore()

    backend = os.getenv("CHAT_MEMORY_BACKEND", "sqlite").strip().lower()
    if backend == "sqlite":
        db_path = os.getenv("CHAT_MEMORY_SQLITE_PATH", "artifacts/memory/chat_memory.sqlite3")
        return SQLiteChatMemoryStore(db_path=db_path)

    raise RuntimeError(f"Unsupported CHAT_MEMORY_BACKEND: {backend}")


@lru_cache(maxsize=1)
def _bootstrap() -> tuple[RehabRAG, VideoRecommender, ChatMemoryStore]:
    index_path = os.getenv("RAG_INDEX_INPUT", "artifacts/rag/bm25_index.pkl")
    canonical_path = os.getenv("RAG_CANONICAL_INPUT", "data/canonical_docs.jsonl")

    if os.path.exists(index_path):
        retriever = BM25Retriever.load(index_path)
    else:
        retriever = BM25Retriever.from_jsonl(canonical_path)
        retriever.save(index_path)

    llm_api = create_vllm_api()
    rag = RehabRAG(
        retriever=retriever,
        llm_api=llm_api,
        candidate_pool=int(os.getenv("RAG_CANDIDATE_POOL", "80")),
        safety_boost=float(os.getenv("RAG_SAFETY_BOOST", "0.08")),
        safety_route=os.getenv("RAG_SAFETY_ROUTE", "1") == "1",
        body_boost=float(os.getenv("RAG_BODY_BOOST", "0.35")),
        body_mismatch_penalty=float(os.getenv("RAG_BODY_MISMATCH_PENALTY", "0.20")),
        body_min_hits=int(os.getenv("RAG_BODY_MIN_HITS", "2")),
    )
    video_recommender = _build_video_recommender()
    memory_store = _build_memory_store()
    return rag, video_recommender, memory_store


app = FastAPI(title="MedGemma Rehab API", version="0.1.0")

cors_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    try:
        rag, _, _ = _bootstrap()
        ok = rag.llm_api.health_check() if rag.llm_api else False
        return {"ok": bool(ok)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        rag, video_recommender, memory_store = _bootstrap()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bootstrap failed: {e}") from e

    if not rag.llm_api or not rag.llm_api.health_check():
        raise HTTPException(status_code=503, detail="vLLM server is unavailable.")

    user_id = req.user_id.strip() if req.user_id and req.user_id.strip() else f"user_{uuid4().hex}"
    session_id = req.session_id.strip() if req.session_id and req.session_id.strip() else f"session_{uuid4().hex}"

    policy_notes: List[str] = []
    history_context: Optional[str] = None
    history_turns_used = 0
    session_turns: List[ChatTurn] = []
    long_term_turns: List[ChatTurn] = []
    effective_query = req.query
    preferred_language: Optional[str] = None
    active_episode_id: Optional[str] = None
    active_episode_slots: dict[str, str] = {}
    force_no_clarify = False
    active_episode = None
    episode_transition_note: Optional[str] = None
    last_assistant_answer: Optional[str] = None

    try:
        session_turns = memory_store.get_recent_session_turns(
            session_id=session_id,
            limit=max(1, int(os.getenv("CHAT_MEMORY_SESSION_LIMIT", "6"))),
        )
        if session_turns:
            last_assistant_answer = session_turns[-1].answer
        if os.getenv("CHAT_USE_LONG_TERM_CONTEXT", "0") == "1":
            long_term_turns = memory_store.get_recent_user_turns(
                user_id=user_id,
                limit=max(1, int(os.getenv("CHAT_MEMORY_LONG_TERM_LIMIT", "4"))),
                exclude_session_id=session_id,
            )
        active_episode = memory_store.get_active_episode(session_id=session_id)
    except Exception:
        policy_notes.append("memory_read_failed")

    use_followup_resolver = os.getenv("CHAT_USE_FOLLOWUP_RESOLVER", "0") == "1"
    if use_followup_resolver:
        follow_up = resolve_follow_up_query(req.query, session_turns[-1] if session_turns else None)
        if follow_up:
            effective_query = follow_up.effective_query
            preferred_language = follow_up.preferred_language
            policy_notes.extend(follow_up.policy_notes)
            if follow_up.mode == "need_option":
                language = preferred_language or detect_query_language(effective_query)
                answer = follow_up.clarify_message or (
                    "Please choose an option letter first." if language == "en" else "請先補上選項字母。"
                )
                body_tags = infer_body_tags(effective_query)
                intent_tags = infer_intent_tags(effective_query)

                try:
                    memory_store.add_turn(
                        user_id=user_id,
                        session_id=session_id,
                        query=req.query,
                        answer=answer,
                        policy_notes=policy_notes,
                        references=[],
                        body_tags=sorted(body_tags),
                        intent_tags=sorted(intent_tags),
                    )
                except Exception:
                    policy_notes.append("memory_write_failed")

                return ChatResponse(
                    user_id=user_id,
                    session_id=session_id,
                    episode_id=active_episode_id,
                    history_turns_used=history_turns_used,
                    effective_query=effective_query,
                    answer=answer,
                    language=language,
                    policy_notes=policy_notes,
                    references=[],
                    episode_slots=active_episode_slots,
                    body_tags=sorted(body_tags),
                    intent_tags=sorted(intent_tags),
                    retrieved_chunks=[],
                    videos=[],
                )

    # Natural-chat fallback: when previous assistant asked a clarification question and
    # current user message is a short follow-up, merge recent user turns into one query
    # so retrieval and slot detection do not reset each turn.
    if session_turns and active_episode and not should_start_new_episode(req.query, active_episode):
        last_turn = session_turns[-1]
        last_assistant_answer = last_turn.answer
        if (
            last_turn.episode_id == active_episode.episode_id
            and _looks_like_clarification_answer(last_turn.answer, last_turn.policy_notes)
            and _looks_like_short_followup(req.query)
        ):
            merged_query = _merge_recent_user_queries(session_turns, req.query, max_queries=3)
            if merged_query and merged_query != req.query:
                effective_query = merged_query
                policy_notes.append("contextual_followup_merge")

    try:
        start_new_episode = should_start_new_episode(req.query, active_episode)
        next_body_bucket = detect_body_bucket(req.query)
        previous_body_bucket = str(active_episode.slots.get("body_bucket", "")).strip() if active_episode else ""
        if start_new_episode and active_episode and previous_body_bucket and next_body_bucket and previous_body_bucket != next_body_bucket:
            episode_transition_note = (
                f"Topic shift detected: previous body={previous_body_bucket}, current body={next_body_bucket}. "
                "Acknowledge the new problem focus. Ask once if the previous issue is already resolved; then continue with current issue only."
            )

        if start_new_episode:
            if active_episode:
                memory_store.close_active_episode(session_id=session_id)
            active_episode = memory_store.start_episode(
                user_id=user_id,
                session_id=session_id,
                summary=req.query[:120],
            )
            policy_notes.append("episode_started")

        if active_episode:
            active_episode_id = active_episode.episode_id
            updates = extract_slot_updates(
                effective_query,
                previous_slots=active_episode.slots,
            )
            if updates:
                active_episode = memory_store.update_episode_slots(
                    episode_id=active_episode.episode_id,
                    updates=updates,
                    source_turn_id=None,
                )
                policy_notes.append("episode_slots_updated")

            active_episode_slots = {k: str(v) for k, v in active_episode.slots.items()}
            force_no_clarify = has_minimum_slots_for_plan(active_episode.slots)
            if force_no_clarify:
                policy_notes.append("slots_sufficient_for_plan")
            slot_context = build_episode_context(active_episode)

            episode_turns = _filter_turns_by_episode(session_turns, active_episode_id)
            history_turns_used = len(episode_turns) + len(long_term_turns)
            history_context = _build_history_context(
                session_turns=episode_turns,
                user_turns=long_term_turns,
                max_chars=max(500, int(os.getenv("CHAT_MEMORY_CONTEXT_MAX_CHARS", "1800"))),
            )
            if slot_context:
                history_context = f"{history_context}\n{slot_context}" if history_context else slot_context
            if episode_transition_note:
                history_context = (
                    f"{history_context}\n{episode_transition_note}" if history_context else episode_transition_note
                )
    except Exception:
        policy_notes.append("episode_memory_failed")

    try:
        result = rag.answer(
            query=effective_query,
            top_k=req.top_k,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            conversation_context=history_context,
            preferred_language=preferred_language,
            force_no_clarify=force_no_clarify,
            known_slots=active_episode_slots,
            last_assistant_answer=last_assistant_answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG answer failed: {e}") from e

    policy_notes.extend(result.policy_notes)
    language = preferred_language or detect_query_language(effective_query)

    rag_body_tags: Set[str] = set()
    for chunk in result.retrieved:
        for tag in chunk.tags:
            if tag.startswith("body_"):
                rag_body_tags.add(tag)

    body_tags = infer_body_tags(effective_query) | rag_body_tags
    intent_tags = infer_intent_tags(effective_query)
    references = _extract_chunk_refs(result.answer)

    videos = video_recommender.recommend(
        query=effective_query,
        body_tags=body_tags,
        intent_tags=intent_tags,
        language=language,
        limit=req.video_limit,
    )

    turn_id: Optional[int] = None
    try:
        turn_id = memory_store.add_turn(
            user_id=user_id,
            session_id=session_id,
            episode_id=active_episode_id,
            query=req.query,
            answer=result.answer,
            policy_notes=policy_notes,
            references=references,
            body_tags=sorted(body_tags),
            intent_tags=sorted(intent_tags),
        )
    except Exception:
        policy_notes.append("memory_write_failed")

    return ChatResponse(
        user_id=user_id,
        session_id=session_id,
        episode_id=active_episode_id,
        history_turns_used=history_turns_used,
        effective_query=effective_query,
        answer=result.answer,
        language=language,
        policy_notes=policy_notes,
        references=references,
        episode_slots=active_episode_slots,
        body_tags=sorted(body_tags),
        intent_tags=sorted(intent_tags),
        retrieved_chunks=[_chunk_to_out(c) for c in result.retrieved],
        videos=[_video_to_out(v, language=language) for v in videos],
    )
