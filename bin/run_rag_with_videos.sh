#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

QUERY="${1:-我背部有點痠痛，請給我建議並推薦可跟著做的影片。}"
SAFETY_ROUTE_FLAG=()
if [[ "${RAG_SAFETY_ROUTE:-1}" == "1" ]]; then
  SAFETY_ROUTE_FLAG+=(--safety-route)
fi

JSON_FLAG=()
if [[ "${RAG_VIDEO_JSON:-0}" == "1" ]]; then
  JSON_FLAG+=(--json)
fi

YT_FLAG=()
if [[ "${VIDEO_USE_YOUTUBE_API:-0}" == "1" ]]; then
  YT_FLAG+=(--use-youtube-api)
fi

YT_KEY_FLAG=()
if [[ -n "${YOUTUBE_API_KEY:-}" ]]; then
  YT_KEY_FLAG+=(--youtube-api-key "${YOUTUBE_API_KEY}")
fi

python scripts/rag_with_videos.py "$QUERY" \
  --index "${RAG_INDEX_INPUT:-artifacts/rag/bm25_index.pkl}" \
  --canonical "${RAG_CANONICAL_INPUT:-data/canonical_docs.jsonl}" \
  --top-k "${RAG_TOP_K:-8}" \
  --candidate-pool "${RAG_CANDIDATE_POOL:-80}" \
  --safety-boost "${RAG_SAFETY_BOOST:-0.08}" \
  --body-boost "${RAG_BODY_BOOST:-0.35}" \
  --body-mismatch-penalty "${RAG_BODY_MISMATCH_PENALTY:-0.20}" \
  --body-min-hits "${RAG_BODY_MIN_HITS:-2}" \
  --temperature "${RAG_TEMPERATURE:-0.0}" \
  --max-tokens "${RAG_MAX_TOKENS:-450}" \
  --video-catalog "${VIDEO_CATALOG:-configs/video_catalog.sample.jsonl}" \
  --video-limit "${VIDEO_LIMIT:-3}" \
  --youtube-region "${YOUTUBE_REGION:-TW}" \
  --youtube-channel-whitelist "${YOUTUBE_CHANNEL_WHITELIST:-}" \
  "${YT_FLAG[@]}" \
  "${YT_KEY_FLAG[@]}" \
  "${SAFETY_ROUTE_FLAG[@]}" \
  "${JSON_FLAG[@]}"
