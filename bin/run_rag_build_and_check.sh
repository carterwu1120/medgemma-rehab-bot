#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Create it first: uv venv .venv && uv pip install -r requirements-preprocess.txt" >&2
  exit 1
fi

source .venv/bin/activate

CANONICAL_OUTPUT="${CANONICAL_OUTPUT:-data/canonical_docs.jsonl}"
COVERAGE_INPUT="${COVERAGE_INPUT:-$CANONICAL_OUTPUT}"

BUILD_ARGS=(
  --input-dir "${PDF_INPUT_DIR:-raw_data/pdfs}"
  --output "$CANONICAL_OUTPUT"
  --chunk-size "${PDF_CHUNK_SIZE:-1200}"
  --overlap "${PDF_CHUNK_OVERLAP:-150}"
  --min-chars "${PDF_MIN_CHARS:-80}"
  --chunk-strategy "${PDF_CHUNK_STRATEGY:-fixed}"
  --min-tag-count "${PDF_MIN_TAG_COUNT:-1}"
  --reject-output "${PDF_REJECT_OUTPUT:-outputs/canonical_rejects.jsonl}"
  --include-files "${PDF_INCLUDE_FILES:-}"
  --exclude-files "${PDF_EXCLUDE_FILES:-}"
)

if [[ "${PDF_ALLOW_GENERAL:-0}" == "1" ]]; then
  BUILD_ARGS+=(--allow-general)
fi

if [[ "${PDF_REQUIRE_BODY_TAG:-0}" == "1" ]]; then
  BUILD_ARGS+=(--require-body-tag)
fi

echo "[1/2] Building canonical RAG docs..."
python scripts/build_canonical_from_pdf.py "${BUILD_ARGS[@]}"

echo "[2/2] Running coverage check..."
python scripts/coverage_check.py \
  --input "$COVERAGE_INPUT" \
  --min-per-body "${COVERAGE_MIN_PER_BODY:-50}" \
  --report-json "${COVERAGE_REPORT_JSON:-outputs/coverage_report.json}"
