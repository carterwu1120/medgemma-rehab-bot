#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

CMD=(
  python scripts/eval_retrieval.py
  --queries "${EVAL_QUERIES:-eval/queries.sample.jsonl}"
  --canonical "${EVAL_CANONICAL:-data/canonical_docs.jsonl}"
  --bm25-index "${EVAL_BM25_INDEX:-artifacts/rag/bm25_index.pkl}"
  --methods "${EVAL_METHODS:-bm25}"
  --top-k "${EVAL_TOP_K:-5}"
  --candidate-pool "${EVAL_CANDIDATE_POOL:-30}"
  --hybrid-alpha "${EVAL_HYBRID_ALPHA:-0.5}"
  --safety-boost "${EVAL_SAFETY_BOOST:-0.0}"
  --bge-dense-model "${EVAL_BGE_DENSE_MODEL:-${EVAL_DENSE_MODEL:-BAAI/bge-m3}}"
  --bge-dense-embeddings "${EVAL_BGE_DENSE_EMBEDDINGS:-${EVAL_DENSE_EMBEDDINGS:-artifacts/rag/bge_m3_dense_only.npy}}"
  --bge-dense-device "${EVAL_BGE_DENSE_DEVICE:-${EVAL_DENSE_DEVICE:-}}"
  --bge-mixed-model "${EVAL_BGE_MIXED_MODEL:-${EVAL_BGE_MODEL:-BAAI/bge-m3}}"
  --bge-mixed-embeddings "${EVAL_BGE_MIXED_EMBEDDINGS:-${EVAL_BGE_EMBEDDINGS:-artifacts/rag/bge_m3_dense_sparse.npy}}"
  --bge-mixed-device "${EVAL_BGE_MIXED_DEVICE:-${EVAL_BGE_DEVICE:-}}"
  --bge-mixed-dense-weight "${EVAL_BGE_MIXED_DENSE_WEIGHT:-${EVAL_BGE_DENSE_WEIGHT:-0.5}}"
  --detailed-output "${EVAL_DETAILED_OUTPUT:-outputs/retrieval_eval_detail.jsonl}"
)

if [[ "${EVAL_SAFETY_ROUTE:-0}" == "1" ]]; then
  CMD+=(--safety-route)
fi

"${CMD[@]}"
