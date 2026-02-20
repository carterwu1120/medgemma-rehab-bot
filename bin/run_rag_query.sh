#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

QUERY="${1:-我運動後小腿和腳踝痠痛，請給我安全的居家復健建議。}"
SAFETY_ROUTE_FLAG=()
if [[ "${RAG_SAFETY_ROUTE:-1}" == "1" ]]; then
  SAFETY_ROUTE_FLAG+=(--safety-route)
fi

if [[ "${RAG_RETRIEVAL_ONLY:-0}" == "1" ]]; then
  python scripts/rag_query.py "$QUERY" \
    --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}" \
    --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
    --top-k "${RAG_TOP_K:-8}" \
    --candidate-pool "${RAG_CANDIDATE_POOL:-80}" \
    --safety-boost "${RAG_SAFETY_BOOST:-0.08}" \
    --body-boost "${RAG_BODY_BOOST:-0.35}" \
    --body-mismatch-penalty "${RAG_BODY_MISMATCH_PENALTY:-0.20}" \
    --body-min-hits "${RAG_BODY_MIN_HITS:-2}" \
    "${SAFETY_ROUTE_FLAG[@]}" \
    --retrieval-only
else
  python scripts/rag_query.py "$QUERY" \
    --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}" \
    --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
    --top-k "${RAG_TOP_K:-8}" \
    --candidate-pool "${RAG_CANDIDATE_POOL:-80}" \
    --safety-boost "${RAG_SAFETY_BOOST:-0.08}" \
    --body-boost "${RAG_BODY_BOOST:-0.35}" \
    --body-mismatch-penalty "${RAG_BODY_MISMATCH_PENALTY:-0.20}" \
    --body-min-hits "${RAG_BODY_MIN_HITS:-2}" \
    "${SAFETY_ROUTE_FLAG[@]}" \
    --temperature "${RAG_TEMPERATURE:-0.2}" \
    --max-tokens "${RAG_MAX_TOKENS:-512}"
fi
