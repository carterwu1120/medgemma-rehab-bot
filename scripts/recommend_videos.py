import argparse
import json
import sys
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.rag import BM25Retriever, RehabRAG
from backend.recommendation import (
    CatalogVideoProvider,
    VideoRecommender,
    YouTubeDataApiProvider,
    YouTubeSearchProvider,
    detect_query_language,
    infer_body_tags,
    infer_intent_tags,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend rehab videos for a user query.")
    parser.add_argument("query", nargs="?", default="我肩頸很緊，請推薦可以跟著做的安全影片")
    parser.add_argument("--catalog", default="configs/video_catalog.sample.jsonl")
    parser.add_argument("--index", default="artifacts/rag/bm25_index.pkl")
    parser.add_argument("--canonical", default="data/canonical_docs.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--retrieve-k", type=int, default=8)
    parser.add_argument("--skip-rag-signals", action="store_true", help="Only infer from raw query, do not use retrieval tags.")
    parser.add_argument("--use-youtube-api", action="store_true")
    parser.add_argument("--youtube-api-key", default="")
    parser.add_argument("--youtube-region", default="TW")
    parser.add_argument("--youtube-channel-whitelist", default="")
    parser.add_argument("--json", action="store_true", help="Output recommendations in JSON.")
    return parser.parse_args()


def enrich_tags_from_retrieval(query: str, index: str, canonical: str, top_k: int) -> Set[str]:
    index_path = Path(index)
    if index_path.exists():
        retriever = BM25Retriever.load(str(index_path))
    else:
        retriever = BM25Retriever.from_jsonl(canonical)
        retriever.save(str(index_path))

    rag = RehabRAG(retriever=retriever, llm_api=None)
    hits = rag.retrieve(query=query, top_k=top_k)

    tags: Set[str] = set()
    for chunk in hits:
        for tag in chunk.tags:
            if tag.startswith("body_"):
                tags.add(tag)
    return tags


def main() -> None:
    args = parse_args()
    query = args.query

    language = detect_query_language(query)
    body_tags = infer_body_tags(query)
    intent_tags = infer_intent_tags(query)

    if not args.skip_rag_signals:
        try:
            body_tags |= enrich_tags_from_retrieval(
                query=query,
                index=args.index,
                canonical=args.canonical,
                top_k=args.retrieve_k,
            )
        except Exception as e:
            print(f"[WARN] Retrieval signal unavailable: {e}", file=sys.stderr)

    providers = [CatalogVideoProvider(args.catalog)]
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

    recommender = VideoRecommender(providers)
    recs = recommender.recommend(
        query=query,
        body_tags=body_tags,
        intent_tags=intent_tags,
        language=language,
        limit=args.limit,
    )

    if args.json:
        payload = {
            "query": query,
            "language": language,
            "body_tags": sorted(body_tags),
            "intent_tags": sorted(intent_tags),
            "recommendations": [r.__dict__ for r in recs],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Video Recommendation")
    print(f"- query: {query}")
    print(f"- language: {language}")
    print(f"- body_tags: {', '.join(sorted(body_tags)) if body_tags else 'none'}")
    print(f"- intent_tags: {', '.join(sorted(intent_tags)) if intent_tags else 'none'}")
    print("")
    for i, r in enumerate(recs, start=1):
        print(f"[{i}] {r.title}")
        print(f"provider={r.provider} score={r.score:.3f} difficulty={r.difficulty}")
        print(f"url={r.url}")
        if r.tags:
            print(f"tags={','.join(r.tags)}")
        if r.why:
            print(f"why={', '.join(r.why)}")
        if r.notes:
            print(f"notes={r.notes}")
        print("")


if __name__ == "__main__":
    main()
