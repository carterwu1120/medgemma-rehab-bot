import argparse
import json
import re
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
SAFETY_ROUTE_SUFFIX = " stop exercise seek medical care red flag 停止運動 就醫 紅旗 立即停止"
SAFETY_QUERY_PATTERNS = [
    re.compile(
        r"\b(stop|urgent|emergency|seek medical care|seek care|see a doctor|go to er|red flag|dizziness|chest pain|numbness|weakness|fever|night pain|should i continue|can i keep training|can i continue)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(停止|就醫|急診|紅旗|胸痛|暈|麻木|無力|發燒|夜間痛|惡化|繼續練|繼續運動|要不要停|是否停止|可不可以繼續|還要繼續嗎|還能繼續嗎|繼續嗎)"),
]
BODY_HINT_PATTERNS = {
    "body_shoulder": re.compile(r"\b(shoulder|rotator cuff|scapula)\b|肩|肩膀|旋轉肌袖", re.IGNORECASE),
    "body_neck_trap": re.compile(r"\b(neck|cervical|trapezius)\b|頸|肩頸|斜方肌", re.IGNORECASE),
    "body_back_spine": re.compile(r"\b(back|lumbar|thoracic|spine|low back)\b|背|腰|脊椎|下背", re.IGNORECASE),
    "body_knee": re.compile(r"\b(knee|patella|meniscus)\b|膝|膝蓋|半月板", re.IGNORECASE),
    "body_ankle_foot": re.compile(r"\b(ankle|foot|achilles|plantar)\b|踝|腳踝|足底|跟腱", re.IGNORECASE),
    "body_hip_glute": re.compile(r"\b(hip|glute|buttock)\b|髖|臀|臀肌", re.IGNORECASE),
    "body_elbow_wrist_hand": re.compile(r"\b(elbow|wrist|forearm|hand)\b|手肘|手腕|前臂|手", re.IGNORECASE),
}
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
    parser.add_argument(
        "--safety-boost",
        type=float,
        default=0.0,
        help="Add score bonus to chunks tagged with safety_* when query has safety intent.",
    )
    parser.add_argument(
        "--safety-route",
        action="store_true",
        help="If safety-intent query has no safety_* chunk in top-k, force-inject best safety chunk from candidates.",
    )

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


def has_safety_intent(query: str) -> bool:
    return any(pattern.search(query) for pattern in SAFETY_QUERY_PATTERNS)


def infer_expected_body_tags(query: str) -> set[str]:
    expected: set[str] = set()
    for tag, pattern in BODY_HINT_PATTERNS.items():
        if pattern.search(query):
            expected.add(tag)
    return expected


def rerank_with_safety_policy(
    hits: List[RetrievedChunk],
    query: str,
    top_k: int,
    safety_boost: float,
    safety_route: bool,
) -> List[RetrievedChunk]:
    if not hits:
        return []

    safety_intent = has_safety_intent(query)
    if not safety_intent:
        return hits[:top_k]

    scored: List[tuple[float, RetrievedChunk]] = []
    for hit in hits:
        boost = safety_boost if (SAFETY_TAGS & set(hit.tags)) else 0.0
        scored.append((float(hit.score) + boost, hit))
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked: List[RetrievedChunk] = [
        RetrievedChunk(
            score=score,
            chunk_id=hit.chunk_id,
            text=hit.text,
            source_name=hit.source_name,
            page=hit.page,
            title=hit.title,
            tags=hit.tags,
        )
        for score, hit in scored
    ]
    top = reranked[:top_k]

    if safety_route and not safety_hit(top):
        safety_candidates = [h for h in reranked if SAFETY_TAGS & set(h.tags)]
        if safety_candidates:
            routed = [safety_candidates[0]]
            routed.extend([h for h in top if h.chunk_id != safety_candidates[0].chunk_id])
            return routed[:top_k]
    return top


def backfill_safety_chunk(
    query: str,
    current_hits: List[RetrievedChunk],
    retriever_bm25: BM25Retriever,
    top_k: int,
    candidate_pool: int,
) -> tuple[List[RetrievedChunk], bool]:
    if safety_hit(current_hits):
        return current_hits, False

    expanded_query = f"{query}{SAFETY_ROUTE_SUFFIX}"
    fallback_hits = retriever_bm25.search(expanded_query, top_k=max(top_k, candidate_pool, 20))
    fallback_safety = [h for h in fallback_hits if SAFETY_TAGS & set(h.tags)]
    if not fallback_safety:
        return current_hits, False

    expected_body = infer_expected_body_tags(query)
    if expected_body:
        body_aligned = [h for h in fallback_safety if expected_body & set(h.tags)]
        best = body_aligned[0] if body_aligned else fallback_safety[0]
    else:
        best = fallback_safety[0]
    merged: List[RetrievedChunk] = [best]
    merged.extend([h for h in current_hits if h.chunk_id != best.chunk_id])
    return merged[:top_k], True


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
    safety_boost: float,
    safety_route: bool,
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
        safety_routed = False
        retrieval_k = max(top_k, candidate_pool) if (safety_boost > 0 or safety_route) else top_k

        if method == "bm25":
            hits = retriever_bm25.search(q.query, top_k=retrieval_k)

        elif method == "bge_m3_dense":
            if retriever_bge_dense is None:
                raise RuntimeError("bge-m3 dense retriever is not initialized.")
            hits = retriever_bge_dense.search(q.query, top_k=retrieval_k)

        elif method == "bge_m3_dense_sparse":
            if retriever_bge_mixed is None:
                raise RuntimeError("bge-m3 dense+sparse retriever is not initialized.")
            hits = retriever_bge_mixed.search(q.query, top_k=retrieval_k)

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
                top_k=max(top_k, candidate_pool),
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
                top_k=max(top_k, candidate_pool),
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        hits = rerank_with_safety_policy(
            hits=hits,
            query=q.query,
            top_k=top_k,
            safety_boost=safety_boost,
            safety_route=safety_route,
        )
        if safety_route and has_safety_intent(q.query):
            hits, safety_routed = backfill_safety_chunk(
                query=q.query,
                current_hits=hits,
                retriever_bm25=retriever_bm25,
                top_k=top_k,
                candidate_pool=candidate_pool,
            )

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
                "safety_routed": safety_routed,
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
            safety_boost=args.safety_boost,
            safety_route=args.safety_route,
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
