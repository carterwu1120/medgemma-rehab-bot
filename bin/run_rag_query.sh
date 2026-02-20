#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

QUERY="${1:-我運動後小腿和腳踝痠痛，請給我安全的居家復健建議。}"

if [[ "${RAG_RETRIEVAL_ONLY:-0}" == "1" ]]; then
  python scripts/rag_query.py "$QUERY" \
    --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}" \
    --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
    --top-k "${RAG_TOP_K:-5}" \
    --retrieval-only
else
  python scripts/rag_query.py "$QUERY" \
    --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}" \
    --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
    --top-k "${RAG_TOP_K:-5}" \
    --temperature "${RAG_TEMPERATURE:-0.2}" \
    --max-tokens "${RAG_MAX_TOKENS:-512}"
fi
