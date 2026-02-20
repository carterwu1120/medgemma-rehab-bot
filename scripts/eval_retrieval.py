import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import BM25Retriever, BGEM3DenseRetriever, BGEM3DenseSparseRetriever, RetrievedChunk


SAFETY_TAGS = {"safety_red_flags", "safety_stop_rules"}
METHOD_ALIASES = {
    "dense": "bge_m3_dense",
    "bge_m3": "bge_m3_dense_sparse",
    "hybrid_dense": "hybrid_bge_m3_dense",
    "hybrid_bge_m3": "hybrid_bge_m3_dense_sparse",
}


@dataclass
class QuerySpec:
    query_id: str
    query: str
    required_tags_any: set[str]
    required_tags_all: set[str]
    gold_chunk_ids: set[str]
    is_safety_query: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval quality for BM25 / bge-m3(dense) / bge-m3(dense+sparse) and hybrids."
        )
    )
    parser.add_argument("--queries", default="eval/queries.sample.jsonl", help="Query spec JSONL path.")
    parser.add_argument("--canonical", default="data/canonical_docs.jsonl")
    parser.add_argument("--bm25-index", default="artifacts/rag/bm25_index.pkl")
    parser.add_argument(
        "--methods",
        default="bm25",
        help=(
            "Comma-separated canonical methods: "
            "bm25,bge_m3_dense,bge_m3_dense_sparse,hybrid_bge_m3_dense,hybrid_bge_m3_dense_sparse"
        ),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-pool", type=int, default=30, help="Candidate size for hybrid merge.")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5, help="BM25 weight in hybrid score.")

    parser.add_argument("--bge-dense-model", default="BAAI/bge-m3")
    parser.add_argument("--bge-dense-embeddings", default="artifacts/rag/bge_m3_dense_only.npy")
    parser.add_argument("--bge-dense-device", default=None)

    parser.add_argument("--bge-mixed-model", default="BAAI/bge-m3")
    parser.add_argument("--bge-mixed-embeddings", default="artifacts/rag/bge_m3_dense_sparse.npy")
    parser.add_argument("--bge-mixed-device", default=None)
    parser.add_argument(
        "--bge-mixed-dense-weight",
        type=float,
        default=0.5,
        help="dense weight in bge-m3 dense+sparse fusion score",
    )

    parser.add_argument("--detailed-output", default="", help="Optional JSONL for per-query details.")
    return parser.parse_args()


def canonicalize_methods(raw_methods: str) -> List[str]:
    methods: List[str] = []
    for m in [x.strip() for x in raw_methods.split(",") if x.strip()]:
        canonical = METHOD_ALIASES.get(m, m)
        if canonical != m:
            print(f"[INFO] method alias '{m}' -> '{canonical}'")
        methods.append(canonical)
    return methods


def load_queries(path: str) -> List[QuerySpec]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Query file not found: {p}")

    out: List[QuerySpec] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                QuerySpec(
                    query_id=str(obj.get("id", "")) or f"q{len(out)+1:03d}",
                    query=str(obj.get("query", "")),
                    required_tags_any=set(obj.get("required_tags_any", []) or []),
                    required_tags_all=set(obj.get("required_tags_all", []) or []),
                    gold_chunk_ids=set(obj.get("gold_chunk_ids", []) or []),
                    is_safety_query=bool(obj.get("is_safety_query", False)),
                )
            )
    return out


def is_relevant(hit: RetrievedChunk, q: QuerySpec) -> bool:
    if q.gold_chunk_ids:
        return hit.chunk_id in q.gold_chunk_ids

    tags = set(hit.tags)
    if q.required_tags_all and not q.required_tags_all.issubset(tags):
        return False
    if q.required_tags_any and not (q.required_tags_any & tags):
        return False

    return bool(q.required_tags_all or q.required_tags_any)


def safety_hit(hits: List[RetrievedChunk]) -> bool:
    return any(bool(SAFETY_TAGS & set(h.tags)) for h in hits)


def normalize_scores(hits: List[RetrievedChunk]) -> Dict[str, float]:
    if not hits:
        return {}
    max_s = max(h.score for h in hits)
    if max_s <= 0:
        return {h.chunk_id: 0.0 for h in hits}
    return {h.chunk_id: h.score / max_s for h in hits}


def combine_hybrid(
    first_hits: List[RetrievedChunk],
    second_hits: List[RetrievedChunk],
    docs_by_chunk: Dict[str, Dict[str, Any]],
    alpha: float,
    top_k: int,
) -> List[RetrievedChunk]:
    first_norm = normalize_scores(first_hits)
    second_norm = normalize_scores(second_hits)
    chunk_ids = set(first_norm.keys()) | set(second_norm.keys())

    merged: List[RetrievedChunk] = []
    for cid in chunk_ids:
        score = alpha * first_norm.get(cid, 0.0) + (1.0 - alpha) * second_norm.get(cid, 0.0)
        d = docs_by_chunk.get(cid)
        if not d:
            continue
        merged.append(
            RetrievedChunk(
                score=score,
                chunk_id=cid,
                text=str(d.get("text", "")),
                source_name=str(d.get("source_name", "")),
                page=int(d.get("page", 0) or 0),
                title=str(d.get("title", "")),
                tags=list(d.get("tags", []) or []),
            )
        )

    merged.sort(key=lambda x: x.score, reverse=True)
    return merged[:top_k]


def evaluate_method(
    method: str,
    queries: List[QuerySpec],
    top_k: int,
    retriever_bm25: BM25Retriever,
    retriever_bge_dense: Optional[BGEM3DenseRetriever],
    retriever_bge_mixed: Optional[BGEM3DenseSparseRetriever],
    candidate_pool: int,
    hybrid_alpha: float,
    docs_by_chunk: Dict[str, Dict[str, Any]],
    detailed_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    hit_count = 0
    rr_sum = 0.0
    safety_total = 0
    safety_hit_count = 0
    latencies: List[float] = []

    for q in queries:
        t0 = time.time()

        if method == "bm25":
            hits = retriever_bm25.search(q.query, top_k=top_k)

        elif method == "bge_m3_dense":
            if retriever_bge_dense is None:
                raise RuntimeError("bge-m3 dense retriever is not initialized.")
            hits = retriever_bge_dense.search(q.query, top_k=top_k)

        elif method == "bge_m3_dense_sparse":
            if retriever_bge_mixed is None:
                raise RuntimeError("bge-m3 dense+sparse retriever is not initialized.")
            hits = retriever_bge_mixed.search(q.query, top_k=top_k)

        elif method == "hybrid_bge_m3_dense":
            if retriever_bge_dense is None:
                raise RuntimeError("bge-m3 dense retriever is not initialized for hybrid_bge_m3_dense.")
            bm25_hits = retriever_bm25.search(q.query, top_k=max(top_k, candidate_pool))
            dense_hits = retriever_bge_dense.search(q.query, top_k=max(top_k, candidate_pool))
            hits = combine_hybrid(
                first_hits=bm25_hits,
                second_hits=dense_hits,
                docs_by_chunk=docs_by_chunk,
                alpha=hybrid_alpha,
                top_k=top_k,
            )

        elif method == "hybrid_bge_m3_dense_sparse":
            if retriever_bge_mixed is None:
                raise RuntimeError(
                    "bge-m3 dense+sparse retriever is not initialized for hybrid_bge_m3_dense_sparse."
                )
            bm25_hits = retriever_bm25.search(q.query, top_k=max(top_k, candidate_pool))
            mixed_hits = retriever_bge_mixed.search(q.query, top_k=max(top_k, candidate_pool))
            hits = combine_hybrid(
                first_hits=bm25_hits,
                second_hits=mixed_hits,
                docs_by_chunk=docs_by_chunk,
                alpha=hybrid_alpha,
                top_k=top_k,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        latencies.append((time.time() - t0) * 1000.0)

        ranks = [i for i, h in enumerate(hits, start=1) if is_relevant(h, q)]
        hit = len(ranks) > 0
        rr = (1.0 / ranks[0]) if ranks else 0.0

        if hit:
            hit_count += 1
        rr_sum += rr

        if q.is_safety_query:
            safety_total += 1
            if safety_hit(hits):
                safety_hit_count += 1

        detailed_rows.append(
            {
                "method": method,
                "query_id": q.query_id,
                "query": q.query,
                "hit": hit,
                "mrr": rr,
                "is_safety_query": q.is_safety_query,
                "safety_hit": safety_hit(hits) if q.is_safety_query else None,
                "top_chunks": [
                    {
                        "rank": i,
                        "chunk_id": h.chunk_id,
                        "score": round(float(h.score), 6),
                        "source_name": h.source_name,
                        "tags": h.tags,
                    }
                    for i, h in enumerate(hits, start=1)
                ],
            }
        )

    n = len(queries)
    return {
        "method": method,
        "queries": n,
        "hit_at_k": hit_count / n if n else 0.0,
        "mrr_at_k": rr_sum / n if n else 0.0,
        "safety_recall": (safety_hit_count / safety_total) if safety_total else None,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "safety_queries": safety_total,
    }


def main() -> None:
    args = parse_args()
    methods = canonicalize_methods(args.methods)
    queries = load_queries(args.queries)
    if not queries:
        raise SystemExit("No queries loaded.")

    bm25_index_path = Path(args.bm25_index)
    if bm25_index_path.exists():
        bm25 = BM25Retriever.load(str(bm25_index_path))
    else:
        bm25 = BM25Retriever.from_jsonl(args.canonical)
        bm25.save(str(bm25_index_path))

    docs_by_chunk = {str(d.get("chunk_id", "")): d for d in bm25.docs}

    bge_dense: Optional[BGEM3DenseRetriever] = None
    need_bge_dense = any(m in {"bge_m3_dense", "hybrid_bge_m3_dense"} for m in methods)
    if need_bge_dense:
        try:
            bge_dense = BGEM3DenseRetriever.from_docs(
                docs=bm25.docs,
                model_name=args.bge_dense_model,
                embeddings_path=args.bge_dense_embeddings,
                device=args.bge_dense_device or None,
            )
        except Exception as e:
            print(f"[WARN] bge-m3 dense retriever unavailable: {e}")

    bge_mixed: Optional[BGEM3DenseSparseRetriever] = None
    need_bge_mixed = any(
        m in {"bge_m3_dense_sparse", "hybrid_bge_m3_dense_sparse"} for m in methods
    )
    if need_bge_mixed:
        try:
            bge_mixed = BGEM3DenseSparseRetriever.from_docs(
                docs=bm25.docs,
                model_name=args.bge_mixed_model,
                embeddings_path=args.bge_mixed_embeddings,
                dense_weight=args.bge_mixed_dense_weight,
                device=args.bge_mixed_device or None,
            )
        except Exception as e:
            print(f"[WARN] bge-m3 dense+sparse retriever unavailable: {e}")

    detailed_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for method in methods:
        if method in {"bge_m3_dense", "hybrid_bge_m3_dense"} and bge_dense is None:
            print(f"[SKIP] {method}: bge-m3 dense retriever unavailable")
            continue
        if method in {"bge_m3_dense_sparse", "hybrid_bge_m3_dense_sparse"} and bge_mixed is None:
            print(f"[SKIP] {method}: bge-m3 dense+sparse retriever unavailable")
            continue

        summary = evaluate_method(
            method=method,
            queries=queries,
            top_k=args.top_k,
            retriever_bm25=bm25,
            retriever_bge_dense=bge_dense,
            retriever_bge_mixed=bge_mixed,
            candidate_pool=args.candidate_pool,
            hybrid_alpha=args.hybrid_alpha,
            docs_by_chunk=docs_by_chunk,
            detailed_rows=detailed_rows,
        )
        summaries.append(summary)

    if not summaries:
        raise SystemExit("No methods were evaluated.")

    print("\nRetrieval Eval Summary")
    print("method                         hit@k   mrr@k   safety_recall   avg_latency_ms")
    for s in summaries:
        safety = "n/a" if s["safety_recall"] is None else f"{s['safety_recall']:.3f}"
        print(
            f"{s['method']:<29} {s['hit_at_k']:.3f}   {s['mrr_at_k']:.3f}   {safety:<13}   {s['avg_latency_ms']:.2f}"
        )

    if args.detailed_output:
        out_path = Path(args.detailed_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in detailed_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nWrote detailed rows: {out_path}")


if __name__ == "__main__":
    main()
