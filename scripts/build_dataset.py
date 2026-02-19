import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset


BODY_PARTS = [
    "back",
    "knee",
    "shoulder",
    "ankle",
    "neck",
    "spine",
    "muscle",
    "joint",
    "waist",
    "disc",
    "foot",
    "heel",
    "hip",
    "elbow",
    "wrist",
    "leg",
    "arm",
    "pain",
    "chronic pain",
]

ACTION_WORDS = [
    "stretching",
    "exercise",
    "physiotherapy",
    "rehabilitation",
    "strengthening",
    "workout",
    "stretch",
    "yoga",
    "mobility",
    "range of motion",
    "home exercise",
    "posture",
]

PURE_RESEARCH_WORDS = [
    "prevalence",
    "training program",
    "curriculum",
    "molecular",
    "gene mapping",
    "genotype",
    "statistically",
    "residents surveyed",
    "education value",
]

SYSTEM_INSTRUCTION = (
    "You are a professional Taiwan home rehabilitation assistant. "
    "Give safe, practical, step-by-step movement advice from the provided information."
)


def extract_qa(entry: Dict, source_name: str) -> Tuple[str, str]:
    if source_name == "MedQuAD":
        question = str(entry.get("question", "") or "")
        answer = str(entry.get("answer", "") or "")
    else:
        question = str(entry.get("input", "") or "")
        answer = str(entry.get("output", "") or "")
    return question, answer


def clean_question(question: str) -> str:
    return (
        question.replace("Doctor, ", "")
        .replace("Hi, ", "")
        .replace("Hello, ", "")
        .strip()
    )


def is_rehab_sample(question: str, answer: str) -> bool:
    if len(question) < 8:
        return False

    text = f"{question} {answer}".lower()
    has_body = any(word in text for word in BODY_PARTS)
    has_action = any(word in text for word in ACTION_WORDS)
    pure_research = any(word in text for word in PURE_RESEARCH_WORDS)
    return has_body and has_action and not pure_research


def filter_and_add(
    dataset: Iterable[Dict], source_name: str, limit: int, output_list: List[Dict]
) -> int:
    count = 0
    for entry in dataset:
        question, answer = extract_qa(entry, source_name)
        question = clean_question(question)

        if not is_rehab_sample(question, answer):
            continue

        output_list.append(
            {
                "source": source_name,
                "instruction": SYSTEM_INSTRUCTION,
                "input": question,
                "raw_output": answer,
            }
        )
        count += 1

        if limit > 0 and count >= limit:
            break

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rehab-focused training JSONL from HF datasets.")
    parser.add_argument("--out", type=str, default="data/rehab_train.jsonl")
    parser.add_argument("--medquad-limit", type=int, default=0)
    parser.add_argument("--chatdoctor-limit", type=int, default=0)
    parser.add_argument("--genmed-limit", type=int, default=0)
    return parser.parse_args()


def write_jsonl(samples: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    combined_samples: List[Dict] = []

    print("Loading MedQuAD...")
    medquad = load_dataset("lavita/MedQuAD", split="train", streaming=True)
    n_medquad = filter_and_add(medquad, "MedQuAD", args.medquad_limit, combined_samples)
    print(f"MedQuAD: {n_medquad}")

    print("Loading ChatDoctor...")
    chatdoctor = load_dataset(
        "lavita/ChatDoctor-HealthCareMagic-100k", split="train", streaming=True
    )
    n_chatdoctor = filter_and_add(
        chatdoctor, "ChatDoctor", args.chatdoctor_limit, combined_samples
    )
    print(f"ChatDoctor: {n_chatdoctor}")

    print("Loading GenMedGPT...")
    genmed = load_dataset("wangrongsheng/GenMedGPT-5k-en", split="train", streaming=True)
    n_genmed = filter_and_add(genmed, "GenMedGPT", args.genmed_limit, combined_samples)
    print(f"GenMedGPT: {n_genmed}")

    out_path = Path(args.out)
    write_jsonl(combined_samples, out_path)
    print(f"Done. Wrote {len(combined_samples)} samples to {out_path}")


if __name__ == "__main__":
    main()
