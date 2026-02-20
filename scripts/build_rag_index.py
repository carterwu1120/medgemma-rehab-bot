import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25 index for RAG retrieval.")
    parser.add_argument(
        "--input",
        default="data/canonical_docs.jsonl",
        help="Canonical JSONL generated from PDF corpus.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/rag/bm25_index.pkl",
        help="Output index path.",
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retriever = BM25Retriever.from_jsonl(canonical_path=args.input, k1=args.k1, b=args.b)
    retriever.save(args.output)
    stats = retriever.stats()

    print(f"Built index: {args.output}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
