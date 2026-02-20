#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

python scripts/build_rag_index.py \
  --input "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
  --output "${RAG_INDEX_OUTPUT:-artifacts/rag/bm25_index.pkl}" \
  --k1 "${RAG_BM25_K1:-1.5}" \
  --b "${RAG_BM25_B:-0.75}"
