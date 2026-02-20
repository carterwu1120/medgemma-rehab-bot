import argparse
import fnmatch
import json
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


REHAB_KEYWORDS = [
    "rehab",
    "rehabilitation",
    "physiotherapy",
    "physical therapy",
    "exercise",
    "stretch",
    "strength",
    "mobility",
    "range of motion",
    "pain",
    "ankle",
    "knee",
    "shoulder",
    "back",
    "neck",
    "spine",
    "lumbar",
    "joint",
    "muscle",
    "gait",
    "balance",
    "復健",
    "物理治療",
    "運動",
    "伸展",
    "肌力",
    "活動度",
    "疼痛",
    "痠痛",
    "肩",
    "頸",
    "背",
    "腰",
    "膝",
    "踝",
]

NOISE_LINE_PATTERNS = [
    r"^how to cite",
    r"\bdoi\s*:",
    r"\bcopyright\b",
    r"all rights reserved",
    r"corresponding author",
    r"\bemail\b",
    r"@",
    r"\baffiliat",
    r"\buniversit",
    r"\breceived\b",
    r"\baccepted\b",
    r"\bpublished\b",
]

NOISE_CHUNK_PATTERNS = [
    r"how to cite",
    r"\bdoi\s*:",
    r"all rights reserved",
    r"\bbibliography\b",
    r"\breferences\b",
]

SECTION_STOP_PATTERN = re.compile(r"^(references|bibliography|acknowledg(e)?ments?)\b", re.IGNORECASE)
NOISE_LINE_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_LINE_PATTERNS]
NOISE_CHUNK_REGEXES = [re.compile(p, re.IGNORECASE) for p in NOISE_CHUNK_PATTERNS]
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[。！？!?;；\.])\s+")

DOSAGE_REGEXES = [
    re.compile(r"\b\d+\s*(sets?|reps?|repetitions?)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*(times?)\s*(per day|daily|per week|weekly)\b", re.IGNORECASE),
    re.compile(r"\bhold\s+for\s+\d+\s*(seconds?|sec|minutes?|min)\b", re.IGNORECASE),
    re.compile(r"每\s*組\s*\d+\s*(次|下)", re.IGNORECASE),
    re.compile(r"每\s*(天|週)\s*\d+\s*次", re.IGNORECASE),
    re.compile(r"維持\s*\d+\s*(秒|分鐘)", re.IGNORECASE),
]

SAFETY_STOP_REGEXES = [
    re.compile(
        r"(pain|soreness|symptoms?).{0,24}(persist|lasting|lasts|more than|over).{0,12}(48|72)\s*(hours?|hrs?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(疼痛|痠痛|症狀).{0,24}(持續|超過|超出).{0,8}(48|72)\s*(小時|hr|hrs|hours)",
        re.IGNORECASE,
    ),
    re.compile(r"(疼痛|痠痛).{0,18}(超過|大於|達到).{0,6}(3|三)\s*(天)", re.IGNORECASE),
]

TAG_RULES = {
    "body_neck_trap": [
        "neck",
        "cervical",
        "trapezius",
        "upper back",
        "levator scapulae",
        "scm",
        "tech neck",
        "頸",
        "肩頸",
        "斜方肌",
        "提肩胛肌",
    ],
    "body_shoulder": ["shoulder", "rotator cuff", "scapula", "deltoid", "肩", "肩膀", "旋轉肌袖", "肩胛"],
    "body_elbow_wrist_hand": [
        "elbow",
        "wrist",
        "forearm",
        "hand",
        "carpal",
        "tennis elbow",
        "golfer",
        "手肘",
        "手腕",
        "前臂",
        "手",
    ],
    "body_back_spine": [
        "back",
        "lumbar",
        "thoracic",
        "spine",
        "low back",
        "背",
        "腰",
        "脊椎",
        "下背",
    ],
    "body_hip_glute": ["hip", "glute", "buttock", "piriformis", "hamstring", "髖", "臀", "臀肌", "梨狀肌", "腿後"],
    "body_knee": ["knee", "patella", "acl", "meniscus", "膝", "膝蓋", "半月板"],
    "body_ankle_foot": ["ankle", "foot", "achilles", "plantar", "heel", "踝", "腳踝", "足底", "跟腱", "腳跟"],
    "goal_pain": ["pain", "analges", "sore", "疼痛", "痠痛", "不適"],
    "goal_mobility": ["mobility", "range of motion", "rom", "flexibility", "活動度", "關節活動", "柔軟度"],
    "goal_strength": ["strength", "resistance", "isometric", "肌力", "阻力", "等長"],
    "context_acute": ["acute", "onset", "first week", "48 hours", "swelling", "sprain", "急性", "扭傷", "腫脹", "受傷初期"],
    "context_chronic": ["chronic", "persistent", "long-term", "over 6 weeks", "慢性", "長期", "反覆"],
    "context_post_exercise": [
        "post-exercise",
        "after exercise",
        "after workout",
        "overuse",
        "training load",
        "doms",
        "delayed onset muscle soreness",
        "return to sport",
        "運動後",
        "訓練後",
        "延遲性痠痛",
        "肌肉痠痛",
        "過度使用",
        "回場",
    ],
    "context_posture": ["posture", "desk", "sitting", "computer", "ergonomic", "forward head", "姿勢", "久坐", "低頭", "電腦"],
    "safety_red_flags": [
        "red flag",
        "fever",
        "incontinence",
        "numbness",
        "progressive weakness",
        "saddle",
        "trauma",
        "emergency",
        "night pain",
        "weight loss",
        "radiating pain",
        "紅旗",
        "發燒",
        "失禁",
        "麻木",
        "無力",
        "夜間痛",
        "放射痛",
        "體重減輕",
    ],
    "safety_stop_rules": [
        "stop",
        "stop exercise",
        "stop exercising",
        "discontinue",
        "cease",
        "pause",
        "worsen",
        "seek medical care",
        "consult doctor",
        "chest pain",
        "dizziness",
        "停止",
        "停止運動",
        "立即停止",
        "症狀惡化",
        "就醫",
        "胸痛",
        "暈眩",
        "pain persists more than 72 hours",
        "pain lasts more than 48 hours",
        "persistent pain after exercise",
        "超過72小時疼痛",
        "超過48小時疼痛",
        "疼痛超過三天",
        "症狀超過三天未改善",
    ],
    "prescription_dosage": [
        "set",
        "sets",
        "rep",
        "reps",
        "repetition",
        "per day",
        "per week",
        "daily",
        "weekly",
        "hold for",
        "每組",
        "每次",
        "每天",
        "每週",
        "秒",
        "分鐘",
    ],
}


def parse_pattern_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def should_keep_file(file_name: str, include_patterns: list[str], exclude_patterns: list[str]) -> bool:
    lowered = file_name.lower()
    if include_patterns and not any(fnmatch.fnmatch(lowered, pat) for pat in include_patterns):
        return False
    if exclude_patterns and any(fnmatch.fnmatch(lowered, pat) for pat in exclude_patterns):
        return False
    return True


def normalize_text(raw: str) -> str:
    text = raw.replace("\x00", " ").replace("\u00ad", "")
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines: list[str] = []

    for line in lines:
        if not line:
            continue
        if SECTION_STOP_PATTERN.match(line):
            break
        if is_noise_line(line):
            continue
        cleaned_lines.append(line)

    text = " ".join(cleaned_lines)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    lowered = line.lower()
    if any(regex.search(lowered) for regex in NOISE_LINE_REGEXES):
        return True

    alpha_count = sum(ch.isalpha() for ch in line)
    digit_count = sum(ch.isdigit() for ch in line)
    total = len(line)

    if total > 0 and alpha_count / total < 0.35 and digit_count / total > 0.15:
        return True

    return False


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = SENTENCE_BOUNDARY_PATTERN.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def chunk_text_fixed(text: str, chunk_size: int, overlap: int, min_chars: int) -> list[str]:
    if not text:
        return []
    step = max(1, chunk_size - overlap)
    chunks: list[str] = []
    for i in range(0, len(text), step):
        chunk = text[i : i + chunk_size].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def chunk_text_sentence(text: str, chunk_size: int, overlap: int, min_chars: int) -> list[str]:
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return chunk_text_fixed(text, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_len = 0

    def flush_current() -> None:
        if not current_sentences:
            return
        chunk = " ".join(current_sentences).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    for sentence in sentences:
        sentence_len = len(sentence) + 1
        if current_sentences and current_len + sentence_len > chunk_size:
            flush_current()

            if overlap > 0:
                overlap_buffer: list[str] = []
                overlap_len = 0
                for prev_sentence in reversed(current_sentences):
                    overlap_buffer.append(prev_sentence)
                    overlap_len += len(prev_sentence) + 1
                    if overlap_len >= overlap:
                        break
                current_sentences = list(reversed(overlap_buffer))
                current_len = sum(len(s) + 1 for s in current_sentences)
            else:
                current_sentences = []
                current_len = 0

        current_sentences.append(sentence)
        current_len += sentence_len

    flush_current()

    if not chunks and len(text) >= min_chars:
        return [text.strip()]
    return chunks


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    min_chars: int,
    strategy: str = "fixed",
) -> list[str]:
    if strategy == "sentence":
        return chunk_text_sentence(text=text, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)
    return chunk_text_fixed(text=text, chunk_size=chunk_size, overlap=overlap, min_chars=min_chars)


def infer_tags(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    for tag, keywords in TAG_RULES.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            tags.append(tag)

    if any(regex.search(text) for regex in DOSAGE_REGEXES):
        tags.append("prescription_dosage")
    if any(regex.search(text) for regex in SAFETY_STOP_REGEXES):
        tags.append("safety_stop_rules")

    return sorted(set(tags))


def evaluate_chunk(
    chunk: str,
    tags: list[str],
    allow_general: bool,
    min_tag_count: int,
    require_body_tag: bool,
) -> tuple[bool, str]:
    lowered = chunk.lower()
    rehab_hits = sum(1 for keyword in REHAB_KEYWORDS if keyword in lowered)
    noise_hits = sum(1 for regex in NOISE_CHUNK_REGEXES if regex.search(lowered))

    alpha_count = sum(ch.isalpha() for ch in chunk)
    digit_count = sum(ch.isdigit() for ch in chunk)
    total = len(chunk)

    if total == 0:
        return False, "empty"

    if noise_hits >= 2 and rehab_hits == 0:
        return False, "noisy_metadata"

    if alpha_count / total < 0.45:
        return False, "low_alpha_ratio"

    if digit_count / max(1, alpha_count + digit_count) > 0.35 and rehab_hits == 0:
        return False, "too_numeric"

    if not allow_general and rehab_hits == 0 and not tags:
        return False, "no_rehab_signal"

    if not allow_general and len(tags) < min_tag_count:
        return False, "insufficient_tags"

    if require_body_tag and not any(tag.startswith("body_") for tag in tags):
        return False, "missing_body_tag"

    return True, "ok"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PDF files into canonical JSONL chunks for RAG indexing."
    )
    parser.add_argument("--input-dir", default="raw_data/pdfs", help="Directory containing PDF files")
    parser.add_argument("--output", default="data/canonical_docs.jsonl", help="Output JSONL path")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap size in characters")
    parser.add_argument("--min-chars", type=int, default=80, help="Minimum chunk length")
    parser.add_argument(
        "--chunk-strategy",
        choices=["fixed", "sentence"],
        default="fixed",
        help="Chunking strategy: fixed-width chars or sentence-aware packing.",
    )
    parser.add_argument(
        "--allow-general",
        action="store_true",
        help="Keep non-rehab chunks. Default is strict rehab-only filtering.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=1,
        help="Minimum inferred tags required in strict mode.",
    )
    parser.add_argument(
        "--require-body-tag",
        action="store_true",
        help="Require at least one body_* tag.",
    )
    parser.add_argument(
        "--reject-output",
        default="outputs/canonical_rejects.jsonl",
        help="Write rejected chunks and reasons to this JSONL path. Use empty string to disable.",
    )
    parser.add_argument(
        "--include-files",
        default="",
        help="Comma-separated glob patterns to include (e.g. '*neck*.pdf,*shoulder*.pdf').",
    )
    parser.add_argument(
        "--exclude-files",
        default="",
        help="Comma-separated glob patterns to exclude.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    include_patterns = parse_pattern_list(args.include_files)
    exclude_patterns = parse_pattern_list(args.exclude_files)

    all_pdf_files = sorted(input_dir.glob("*.pdf"))
    pdf_files = [
        p for p in all_pdf_files if should_keep_file(p.name, include_patterns, exclude_patterns)
    ]

    if not pdf_files:
        raise SystemExit(f"No PDF files found in {input_dir} after include/exclude filtering")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    reject_path = Path(args.reject_output) if args.reject_output else None
    if reject_path:
        reject_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0
    skipped_chunks = 0
    reject_reasons: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as out:
        reject_file = reject_path.open("w", encoding="utf-8") if reject_path else None
        try:
            for pdf_path in pdf_files:
                reader = PdfReader(str(pdf_path))
                doc_id = pdf_path.stem.replace(" ", "_")
                title = pdf_path.stem

                for page_num, page in enumerate(reader.pages, start=1):
                    raw = page.extract_text() or ""
                    text = normalize_text(raw)
                    chunks = chunk_text(
                        text=text,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                        min_chars=args.min_chars,
                        strategy=args.chunk_strategy,
                    )

                    kept_idx = 0
                    for raw_idx, chunk in enumerate(chunks, start=1):
                        tags = infer_tags(chunk)
                        keep, reason = evaluate_chunk(
                            chunk=chunk,
                            tags=tags,
                            allow_general=args.allow_general,
                            min_tag_count=max(0, args.min_tag_count),
                            require_body_tag=args.require_body_tag,
                        )

                        if not keep:
                            skipped_chunks += 1
                            reject_reasons[reason] += 1
                            if reject_file:
                                reject_record = {
                                    "doc_id": doc_id,
                                    "source_name": pdf_path.name,
                                    "page": page_num,
                                    "raw_chunk_index": raw_idx,
                                    "reason": reason,
                                    "tags": tags,
                                    "text": chunk,
                                }
                                reject_file.write(json.dumps(reject_record, ensure_ascii=False) + "\n")
                            continue

                        kept_idx += 1
                        record = {
                            "doc_id": doc_id,
                            "source_type": "pdf",
                            "source_name": pdf_path.name,
                            "title": title,
                            "page": page_num,
                            "chunk_id": f"{doc_id}_p{page_num}_c{kept_idx}",
                            "text": chunk,
                            "tags": tags,
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_chunks += 1

                total_docs += 1
        finally:
            if reject_file:
                reject_file.close()

    skipped_files = len(all_pdf_files) - len(pdf_files)
    reason_str = ", ".join(f"{k}:{v}" for k, v in reject_reasons.most_common()) or "none"
    print(
        f"Done. docs={total_docs}, chunks={total_chunks}, skipped_chunks={skipped_chunks}, "
        f"reject_reasons={reason_str}, skipped_files={skipped_files}, "
        f"chunk_strategy={args.chunk_strategy}, output={output_path}"
    )
    if reject_path:
        print(f"Reject log: {reject_path}")


if __name__ == "__main__":
    main()
