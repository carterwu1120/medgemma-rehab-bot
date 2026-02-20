import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


BODY_BUCKETS = {
    "neck_trap": [
        "neck",
        "cervical",
        "trapezius",
        "trap",
        "upper back",
    ],
    "shoulder": [
        "shoulder",
        "rotator cuff",
        "scapula",
        "deltoid",
    ],
    "elbow_wrist_hand": [
        "elbow",
        "wrist",
        "forearm",
        "hand",
        "carpal",
        "tennis elbow",
        "golfer",
    ],
    "back_spine": [
        "back",
        "lumbar",
        "thoracic",
        "spine",
        "low back",
    ],
    "hip_glute": [
        "hip",
        "glute",
        "buttock",
        "piriformis",
        "hamstring",
    ],
    "knee": [
        "knee",
        "patella",
        "acl",
        "meniscus",
    ],
    "ankle_foot": [
        "ankle",
        "foot",
        "achilles",
        "plantar",
        "heel",
    ],
}

CONTEXT_BUCKETS = {
    "acute": ["acute", "onset", "first week", "48 hours", "swelling", "sprain"],
    "chronic": ["chronic", "persistent", "long-term", "over 6 weeks"],
    "post_exercise": ["post-exercise", "after exercise", "overuse", "training load"],
    "posture": ["posture", "desk", "sitting", "computer", "ergonomic", "forward head"],
}

SAFETY_BUCKETS = {
    "red_flags": [
        "red flag",
        "fever",
        "incontinence",
        "numbness",
        "progressive weakness",
        "saddle",
        "trauma",
        "urgent",
        "emergency",
    ],
    "stop_rules": [
        "stop",
        "discontinue",
        "worsen",
        "if pain increases",
        "seek medical care",
    ],
}

DOSAGE_PATTERNS = [
    re.compile(r"\b\d+\s*(sets?|reps?|repetitions?)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(times?)\s*(per day|daily|per week|weekly)\b", re.IGNORECASE),
    re.compile(r"\bhold\s+for\s+\d+\s*(seconds?|sec|minutes?|min)\b", re.IGNORECASE),
]

PROGRESSION_KEYWORDS = [
    "progress",
    "regress",
    "advance",
    "increase load",
    "decrease load",
    "next phase",
    "if tolerated",
]


def contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def has_dosage_pattern(text: str) -> bool:
    return any(pattern.search(text) for pattern in DOSAGE_PATTERNS)


def is_chinese_text(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage report for canonical RAG chunks.")
    parser.add_argument("--input", default="data/canonical_docs.jsonl", help="Input canonical JSONL")
    parser.add_argument("--min-per-body", type=int, default=50, help="Target minimum chunks per body bucket")
    parser.add_argument("--report-json", default="", help="Optional output path for JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    if total == 0:
        raise SystemExit("Input file is empty.")

    body_counts = {key: 0 for key in BODY_BUCKETS}
    context_counts = {key: 0 for key in CONTEXT_BUCKETS}
    safety_counts = {key: 0 for key in SAFETY_BUCKETS}
    matrix = {body: {ctx: 0 for ctx in CONTEXT_BUCKETS} for body in BODY_BUCKETS}

    dosage_count = 0
    progression_count = 0
    zh_count = 0

    for row in rows:
        text = str(row.get("text", "")).lower()

        body_hits = []
        for body, keywords in BODY_BUCKETS.items():
            if contains_any(text, keywords):
                body_counts[body] += 1
                body_hits.append(body)

        context_hits = []
        for context, keywords in CONTEXT_BUCKETS.items():
            if contains_any(text, keywords):
                context_counts[context] += 1
                context_hits.append(context)

        for safety, keywords in SAFETY_BUCKETS.items():
            if contains_any(text, keywords):
                safety_counts[safety] += 1

        if has_dosage_pattern(text):
            dosage_count += 1

        if contains_any(text, PROGRESSION_KEYWORDS):
            progression_count += 1

        if is_chinese_text(str(row.get("text", ""))):
            zh_count += 1

        for body in body_hits:
            for context in context_hits:
                matrix[body][context] += 1

    print(f"Total chunks: {total}")
    print("\nBody Coverage:")
    for body, count in body_counts.items():
        pct = (count / total) * 100
        status = "OK" if count >= args.min_per_body else "LOW"
        print(f"- {body:16s} {count:4d} ({pct:5.1f}%) [{status}]")

    print("\nContext Coverage:")
    for context, count in context_counts.items():
        pct = (count / total) * 100
        print(f"- {context:16s} {count:4d} ({pct:5.1f}%)")

    print("\nSafety & Prescription:")
    print(f"- red_flags        {safety_counts['red_flags']:4d} ({(safety_counts['red_flags']/total)*100:5.1f}%)")
    print(f"- stop_rules       {safety_counts['stop_rules']:4d} ({(safety_counts['stop_rules']/total)*100:5.1f}%)")
    print(f"- dosage_pattern   {dosage_count:4d} ({(dosage_count/total)*100:5.1f}%)")
    print(f"- progression      {progression_count:4d} ({(progression_count/total)*100:5.1f}%)")

    print("\nLanguage:")
    print(f"- chinese_text     {zh_count:4d} ({(zh_count/total)*100:5.1f}%)")

    print("\nBody x Context Matrix (hit counts):")
    header = "body\\context".ljust(16) + " " + " ".join(f"{c:>10s}" for c in CONTEXT_BUCKETS)
    print(header)
    for body, ctx_counts in matrix.items():
        row = body.ljust(16) + " " + " ".join(f"{ctx_counts[c]:10d}" for c in CONTEXT_BUCKETS)
        print(row)

    low_bodies = [body for body, count in body_counts.items() if count < args.min_per_body]
    if low_bodies:
        print("\nGaps to fill first:")
        for body in low_bodies:
            print(f"- {body} (target >= {args.min_per_body}, current {body_counts[body]})")

    if args.report_json:
        report = {
            "total_chunks": total,
            "body_counts": body_counts,
            "context_counts": context_counts,
            "safety_counts": safety_counts,
            "dosage_count": dosage_count,
            "progression_count": progression_count,
            "chinese_count": zh_count,
            "matrix": matrix,
            "target_min_per_body": args.min_per_body,
            "low_bodies": low_bodies,
        }
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()
