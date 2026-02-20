import argparse
import json
import sys
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_api import create_vllm_api
from backend.rag import BM25Retriever, RehabRAG
from backend.recommendation import (
    CatalogVideoProvider,
    VideoRecommender,
    YouTubeDataApiProvider,
    YouTubeSearchProvider,
    detect_query_language,
    infer_body_tags,
    infer_intent_tags,
    summarize_video,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG answer + recommended rehab videos with brief summaries.")
    parser.add_argument("query", nargs="?", default="我背部有點痠痛，請給我建議並推薦可跟著做的影片。")
    parser.add_argument("--index", default="artifacts/rag/bm25_index.pkl")
    parser.add_argument("--canonical", default="data/canonical_docs.jsonl")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=450)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--safety-boost", type=float, default=0.08)
    parser.add_argument("--safety-route", action="store_true")
    parser.add_argument("--body-boost", type=float, default=0.35)
    parser.add_argument("--body-mismatch-penalty", type=float, default=0.20)
    parser.add_argument("--body-min-hits", type=int, default=2)
    parser.add_argument("--video-catalog", default="configs/video_catalog.sample.jsonl")
    parser.add_argument("--video-limit", type=int, default=3)
    parser.add_argument("--use-youtube-api", action="store_true")
    parser.add_argument("--youtube-api-key", default="")
    parser.add_argument("--youtube-region", default="TW")
    parser.add_argument("--youtube-channel-whitelist", default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def ensure_retriever(index_path: str, canonical_path: str) -> BM25Retriever:
    path = Path(index_path)
    if path.exists():
        return BM25Retriever.load(str(path))
    retriever = BM25Retriever.from_jsonl(canonical_path)
    retriever.save(str(path))
    return retriever


def build_video_context(query: str, rag_tags: Set[str]) -> tuple[str, Set[str], Set[str]]:
    language = detect_query_language(query)
    body_tags = infer_body_tags(query) | rag_tags
    intent_tags = infer_intent_tags(query)
    return language, body_tags, intent_tags


def main() -> None:
    args = parse_args()

    retriever = ensure_retriever(args.index, args.canonical)
    llm_api = create_vllm_api()

    rag = RehabRAG(
        retriever=retriever,
        llm_api=llm_api,
        candidate_pool=args.candidate_pool,
        safety_boost=args.safety_boost,
        safety_route=args.safety_route,
        body_boost=args.body_boost,
        body_mismatch_penalty=args.body_mismatch_penalty,
        body_min_hits=args.body_min_hits,
    )

    result = rag.answer(
        query=args.query,
        top_k=args.top_k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    rag_body_tags: Set[str] = set()
    for chunk in result.retrieved:
        for tag in chunk.tags:
            if tag.startswith("body_"):
                rag_body_tags.add(tag)

    language, body_tags, intent_tags = build_video_context(args.query, rag_body_tags)
    providers = [CatalogVideoProvider(args.video_catalog)]
    channel_whitelist = [x.strip() for x in args.youtube_channel_whitelist.split(",") if x.strip()]
    yt_api = YouTubeDataApiProvider(
        api_key=args.youtube_api_key or None,
        region_code=args.youtube_region,
        channel_whitelist=channel_whitelist,
    )
    if args.use_youtube_api and yt_api.enabled:
        providers.append(yt_api)
    else:
        if args.use_youtube_api and not yt_api.enabled:
            print("[WARN] --use-youtube-api is set but YOUTUBE_API_KEY is missing; fallback to search links.", file=sys.stderr)
        providers.append(YouTubeSearchProvider())

    recommender = VideoRecommender(providers=providers)
    videos = recommender.recommend(
        query=args.query,
        body_tags=body_tags,
        intent_tags=intent_tags,
        language=language,
        limit=args.video_limit,
    )

    if args.json:
        payload = {
            "query": args.query,
            "language": language,
            "answer": result.answer,
            "policy_notes": result.policy_notes,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in result.retrieved],
            "videos": [
                {
                    **video.__dict__,
                    "summary": summarize_video(video, language=language),
                }
                for video in videos
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("=== RAG Answer ===\n")
    print(result.answer)
    if result.policy_notes:
        print("\n=== Policy Notes ===")
        print(", ".join(result.policy_notes))

    print("\n=== Recommended Videos ===")
    for i, video in enumerate(videos, start=1):
        print(f"\n[{i}] {video.title}")
        print(f"url={video.url}")
        print(f"provider={video.provider} score={video.score:.3f}")
        print(f"summary={summarize_video(video, language=language)}")
        if video.why:
            print(f"why={', '.join(video.why)}")


if __name__ == "__main__":
    main()
