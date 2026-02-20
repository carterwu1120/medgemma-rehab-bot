#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

QUERY="${1:-我肩頸很緊，請推薦可以跟著做的安全影片}"

ARGS=(
  "$QUERY"
  --catalog "${VIDEO_CATALOG:-configs/video_catalog.sample.jsonl}"
  --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}"
  --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}"
  --limit "${VIDEO_LIMIT:-5}"
  --retrieve-k "${VIDEO_RETRIEVE_K:-8}"
  --youtube-region "${YOUTUBE_REGION:-TW}"
  --youtube-channel-whitelist "${YOUTUBE_CHANNEL_WHITELIST:-}"
)

if [[ "${VIDEO_SKIP_RAG_SIGNALS:-0}" == "1" ]]; then
  ARGS+=(--skip-rag-signals)
fi

if [[ "${VIDEO_USE_YOUTUBE_API:-0}" == "1" ]]; then
  ARGS+=(--use-youtube-api)
fi

if [[ -n "${YOUTUBE_API_KEY:-}" ]]; then
  ARGS+=(--youtube-api-key "${YOUTUBE_API_KEY}")
fi

if [[ "${VIDEO_OUTPUT_JSON:-0}" == "1" ]]; then
  ARGS+=(--json)
fi

python scripts/recommend_videos.py "${ARGS[@]}"
