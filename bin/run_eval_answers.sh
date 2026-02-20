#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

SAFETY_ROUTE_FLAG=()
if [[ "${ANSWER_EVAL_SAFETY_ROUTE:-1}" == "1" ]]; then
  SAFETY_ROUTE_FLAG+=(--safety-route)
fi

python scripts/eval_answers.py \
  --eval-set "${ANSWER_EVAL_SET:-eval/answer_eval_set.jsonl}" \
  --index "${ANSWER_EVAL_INDEX:-artifacts/rag/bm25_index.pkl}" \
  --canonical "${ANSWER_EVAL_CANONICAL:-data/canonical_docs.jsonl}" \
  --top-k "${ANSWER_EVAL_TOP_K:-8}" \
  --temperature "${ANSWER_EVAL_TEMPERATURE:-0.0}" \
  --max-tokens "${ANSWER_EVAL_MAX_TOKENS:-450}" \
  --candidate-pool "${ANSWER_EVAL_CANDIDATE_POOL:-80}" \
  --safety-boost "${ANSWER_EVAL_SAFETY_BOOST:-0.08}" \
  --body-boost "${ANSWER_EVAL_BODY_BOOST:-0.35}" \
  --body-mismatch-penalty "${ANSWER_EVAL_BODY_MISMATCH_PENALTY:-0.20}" \
  --body-min-hits "${ANSWER_EVAL_BODY_MIN_HITS:-2}" \
  --limit "${ANSWER_EVAL_LIMIT:-0}" \
  --report-json "${ANSWER_EVAL_REPORT_JSON:-outputs/answer_eval_report.json}" \
  --details-jsonl "${ANSWER_EVAL_DETAILS_JSONL:-outputs/answer_eval_rows.jsonl}" \
  "${SAFETY_ROUTE_FLAG[@]}"
