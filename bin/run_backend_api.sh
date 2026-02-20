#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt -r requirements-api.txt" >&2
  exit 1
fi

source .venv/bin/activate

HOST="${BACKEND_HOST:-0.0.0.0}"
PORT="${BACKEND_PORT:-9000}"

if [[ "${BACKEND_RELOAD:-0}" == "1" ]]; then
  exec uvicorn backend.api.server:app --host "$HOST" --port "$PORT" --reload
fi

exec uvicorn backend.api.server:app --host "$HOST" --port "$PORT"
