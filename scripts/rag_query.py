import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_api import create_vllm_api
from backend.rag import BM25Retriever, RehabRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval or full RAG answer with local BM25 index.")
    parser.add_argument("query", nargs="?", default="我肩膀和頸部很緊，請給我安全的居家復健建議。")
    parser.add_argument("--index", default="artifacts/rag/bm25_index.pkl")
    parser.add_argument("--canonical", default="data/canonical_docs.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--safety-boost", type=float, default=0.08)
    parser.add_argument("--safety-route", action="store_true")
    parser.add_argument("--body-boost", type=float, default=0.35)
    parser.add_argument("--body-mismatch-penalty", type=float, default=0.20)
    parser.add_argument("--body-min-hits", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    index_path = Path(args.index)
    if not index_path.exists():
        print(f"Index not found at {index_path}. Building from {args.canonical} ...")
        retriever = BM25Retriever.from_jsonl(args.canonical)
        retriever.save(str(index_path))
    else:
        retriever = BM25Retriever.load(str(index_path))

    rag = RehabRAG(
        retriever=retriever,
        llm_api=None,
        candidate_pool=args.candidate_pool,
        safety_boost=args.safety_boost,
        safety_route=args.safety_route,
        body_boost=args.body_boost,
        body_mismatch_penalty=args.body_mismatch_penalty,
        body_min_hits=args.body_min_hits,
    )

    if args.retrieval_only:
        hits = rag.retrieve(args.query, top_k=args.top_k)
        print(f"Retrieved {len(hits)} chunks")
        for i, h in enumerate(hits, start=1):
            print(f"\n[{i}] score={h.score:.4f} source={h.source_name} p{h.page} chunk={h.chunk_id}")
            print(f"tags={','.join(h.tags) if h.tags else 'none'}")
            print(h.text[:500])
        return

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

    print("=== Retrieved Chunks ===")
    for i, h in enumerate(result.retrieved, start=1):
        print(f"[{i}] score={h.score:.4f} source={h.source_name} p{h.page} chunk={h.chunk_id}")

    print("\n=== Answer ===\n")
    print(result.answer)
    if result.policy_notes:
        print("\n=== Retrieval Policy Notes ===")
        print(", ".join(result.policy_notes))


if __name__ == "__main__":
    main()
