import os
import re
from functools import lru_cache
from typing import List, Optional, Set

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm_api import create_vllm_api
from rag import BM25Retriever, RehabRAG, RetrievedChunk
from recommendation import (
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


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)
    video_limit: int = Field(default=3, ge=0, le=10)
    temperature: float = Field(default=0.0, ge=0.0, le=1.5)
    max_tokens: int = Field(default=450, ge=64, le=2048)


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
    answer: str
    language: str
    policy_notes: List[str]
    references: List[str]
    body_tags: List[str]
    intent_tags: List[str]
    retrieved_chunks: List[RetrievedChunkOut]
    videos: List[VideoOut]


def _extract_chunk_refs(text: str) -> List[str]:
    return sorted(set(CHUNK_ID_PATTERN.findall(text)))


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


@lru_cache(maxsize=1)
def _bootstrap() -> tuple[RehabRAG, VideoRecommender]:
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
    return rag, video_recommender


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
        rag, _ = _bootstrap()
        ok = rag.llm_api.health_check() if rag.llm_api else False
        return {"ok": bool(ok)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        rag, video_recommender = _bootstrap()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bootstrap failed: {e}") from e

    if not rag.llm_api or not rag.llm_api.health_check():
        raise HTTPException(status_code=503, detail="vLLM server is unavailable.")

    try:
        result = rag.answer(
            query=req.query,
            top_k=req.top_k,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG answer failed: {e}") from e

    language = detect_query_language(req.query)
    rag_body_tags: Set[str] = set()
    for chunk in result.retrieved:
        for tag in chunk.tags:
            if tag.startswith("body_"):
                rag_body_tags.add(tag)

    body_tags = infer_body_tags(req.query) | rag_body_tags
    intent_tags = infer_intent_tags(req.query)
    videos = video_recommender.recommend(
        query=req.query,
        body_tags=body_tags,
        intent_tags=intent_tags,
        language=language,
        limit=req.video_limit,
    )

    return ChatResponse(
        answer=result.answer,
        language=language,
        policy_notes=result.policy_notes,
        references=_extract_chunk_refs(result.answer),
        body_tags=sorted(body_tags),
        intent_tags=sorted(intent_tags),
        retrieved_chunks=[_chunk_to_out(c) for c in result.retrieved],
        videos=[_video_to_out(v, language=language) for v in videos],
    )

