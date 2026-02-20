import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_api import create_vllm_api
from rag import BM25Retriever, RehabRAG


# chunk_id often contains '-' in source-derived prefixes.
CHUNK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+_p\d+_c\d+")
STEP_PATTERN = re.compile(r"(^|\n)\s*(\d+[\.\)]|[-*])\s+", re.MULTILINE)
DOSAGE_PATTERN = re.compile(
    r"\b(\d+\s*(sets?|reps?|times?|minutes?|mins?|sec|seconds?))\b|"
    r"(每\s*(天|週)\s*\d+\s*次)|(每組\s*\d+\s*次)|(分鐘|秒)",
    re.IGNORECASE,
)
STOP_PATTERN = re.compile(
    r"\b(stop|discontinue|pause|worse|worsen|halt|hold)\b|停止|停訓|停练|惡化|恶化|中止",
    re.IGNORECASE,
)
RED_FLAG_PATTERN = re.compile(
    r"\b(red flag|numbness|weakness|fever|chest pain|dizziness|night pain)\b|紅旗|麻木|無力|无力|發燒|发烧|胸痛|暈|晕|夜間痛|夜间痛",
    re.IGNORECASE,
)
SEEK_CARE_PATTERN = re.compile(
    r"\b(seek medical care|see a doctor|emergency|go to er|medical attention)\b|就醫|就医|急診|急诊|看醫生|看医生",
    re.IGNORECASE,
)
PROGRESSION_PATTERN = re.compile(r"\b(progress|gradual|increase|phase)\b|漸進|進階|逐步|階段", re.IGNORECASE)
DO_DONT_PATTERN = re.compile(r"\b(do|don't|avoid)\b|可以做|不能做|避免", re.IGNORECASE)
DIFF_PATTERN = re.compile(r"\b(vs|difference|distinguish)\b|如何判斷|差異|分辨", re.IGNORECASE)
DECISION_PATTERN = re.compile(
    r"\b(continue|reduce|stop)\b.*\b(continue|reduce|stop)\b|繼續.*降強度.*停訓|決策|判準",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate generated RAG answers with heuristic checks.")
    parser.add_argument("--eval-set", default="eval/answer_eval_set.jsonl")
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
    parser.add_argument("--limit", type=int, default=0, help="Only evaluate first N rows; 0 means all.")
    parser.add_argument("--report-json", default="outputs/answer_eval_report.json")
    parser.add_argument("--details-jsonl", default="outputs/answer_eval_rows.jsonl")
    return parser.parse_args()


def load_eval_set(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Eval set not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def extract_chunk_refs(answer: str) -> List[str]:
    return sorted(set(CHUNK_ID_PATTERN.findall(answer)))


def check_rule(name: str, answer: str, refs: List[str], retrieved_chunk_ids: set[str]) -> bool:
    lowered = answer.lower()
    if name == "chunk_citations":
        return bool(refs) and any(ref in retrieved_chunk_ids for ref in refs)
    if name == "step_by_step":
        return bool(STEP_PATTERN.search(answer))
    if name in {"frequency_or_reps", "dosage", "phase_criteria"}:
        return bool(DOSAGE_PATTERN.search(answer))
    if name in {"stop_condition"}:
        return bool(STOP_PATTERN.search(answer))
    if name in {"red_flags"}:
        return bool(RED_FLAG_PATTERN.search(answer))
    if name in {"when_to_seek_care", "seek_care"}:
        return bool(SEEK_CARE_PATTERN.search(answer))
    if name in {"progression", "return_criteria", "load_modification", "routine_plan", "two_week_plan"}:
        return bool(PROGRESSION_PATTERN.search(answer) or DOSAGE_PATTERN.search(answer))
    if name in {"do_and_dont"}:
        return bool(DO_DONT_PATTERN.search(answer))
    if name in {"differential_guidance"}:
        return bool(DIFF_PATTERN.search(answer))
    if name in {"decision_framework"}:
        return bool(DECISION_PATTERN.search(answer))
    return bool(lowered.strip())


def main() -> None:
    args = parse_args()
    eval_rows = load_eval_set(args.eval_set, limit=args.limit)
    if not eval_rows:
        raise SystemExit("No eval rows loaded.")

    index_path = Path(args.index)
    if index_path.exists():
        retriever = BM25Retriever.load(str(index_path))
    else:
        retriever = BM25Retriever.from_jsonl(args.canonical)
        retriever.save(str(index_path))

    llm_api = create_vllm_api()
    if not llm_api.health_check():
        raise SystemExit("vLLM server is not reachable. Start it first: python scripts/vllm_server.py")

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

    details: List[Dict[str, Any]] = []
    pass_rates: List[float] = []
    safety_pass_rates: List[float] = []

    for row in eval_rows:
        query = str(row.get("query", ""))
        must_have = list(row.get("must_have", []) or [])
        result = rag.answer(
            query=query,
            top_k=args.top_k,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        answer = result.answer
        refs = extract_chunk_refs(answer)
        retrieved_ids = {chunk.chunk_id for chunk in result.retrieved}

        checks: Dict[str, bool] = {}
        for rule in must_have:
            checks[rule] = check_rule(rule, answer, refs, retrieved_ids)

        passed = sum(1 for ok in checks.values() if ok)
        total = max(1, len(must_have))
        pass_rate = passed / total
        pass_rates.append(pass_rate)
        if bool(row.get("is_safety_query", False)):
            safety_pass_rates.append(pass_rate)

        details.append(
            {
                "id": row.get("id"),
                "query": query,
                "is_safety_query": bool(row.get("is_safety_query", False)),
                "must_have": must_have,
                "checks": checks,
                "pass_rate": round(pass_rate, 4),
                "policy_notes": result.policy_notes,
                "retrieved_chunk_ids": [chunk.chunk_id for chunk in result.retrieved],
                "references_in_answer": refs,
                "answer": answer,
            }
        )

    summary = {
        "num_queries": len(details),
        "avg_pass_rate": round(sum(pass_rates) / max(1, len(pass_rates)), 4),
        "avg_safety_pass_rate": round(sum(safety_pass_rates) / max(1, len(safety_pass_rates)), 4)
        if safety_pass_rates
        else None,
        "config": {
            "top_k": args.top_k,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "candidate_pool": args.candidate_pool,
            "safety_boost": args.safety_boost,
            "safety_route": args.safety_route,
            "body_boost": args.body_boost,
            "body_mismatch_penalty": args.body_mismatch_penalty,
            "body_min_hits": args.body_min_hits,
        },
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"summary": summary, "rows": details}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    details_path = Path(args.details_jsonl)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    with details_path.open("w", encoding="utf-8") as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Answer Eval Summary")
    print(f"- queries: {summary['num_queries']}")
    print(f"- avg_pass_rate: {summary['avg_pass_rate']:.4f}")
    if summary["avg_safety_pass_rate"] is not None:
        print(f"- avg_safety_pass_rate: {summary['avg_safety_pass_rate']:.4f}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote details: {details_path}")


if __name__ == "__main__":
    main()
