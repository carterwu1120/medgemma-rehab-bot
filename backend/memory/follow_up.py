from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from .store import ChatTurn


QUESTION_LINE_PATTERN = re.compile(r"^\s*([1-3])[)\.]?\s*(.+?)\s*$")
OPTION_MARKER_PATTERN = re.compile(r"([A-Da-d])\s*[)）.:：-]")
QUESTION_NUMBER_PATTERN = re.compile(r"(?:^|\b|第)\s*([1-3])\s*(?:題|question)?", re.IGNORECASE)
OPTION_LETTER_PATTERN = re.compile(r"\b([A-Da-d])\b")
SHORT_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:如果是|if|選|option|答案|answer|第?\s*[1-3]\s*題?)?\s*[1-3]?\s*[A-Da-d]?\s*$",
    re.IGNORECASE,
)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@dataclass(frozen=True)
class FollowUpResolution:
    effective_query: str
    policy_notes: list[str]
    preferred_language: Optional[str] = None
    mode: str = "expanded"
    clarify_message: Optional[str] = None


def _guess_language(text: str) -> str:
    return "zh" if CJK_PATTERN.search(text) else "en"


def _extract_options(question_line: str) -> Dict[str, str]:
    option_source = question_line
    bracket_match = re.search(r"[（(]([^()（）]+)[)）]\s*$", question_line)
    if bracket_match:
        option_source = bracket_match.group(1)

    markers = list(OPTION_MARKER_PATTERN.finditer(option_source))
    options: Dict[str, str] = {}
    if not markers:
        return options

    for idx, marker in enumerate(markers):
        letter = marker.group(1).upper()
        start = marker.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(option_source)
        cleaned = option_source[start:end].strip(" ,，。；;:：！？!?")
        if cleaned:
            options[letter] = cleaned
    return options


def _parse_clarification_questions(answer: str) -> dict[int, dict[str, str]]:
    questions: dict[int, dict[str, str]] = {}
    max_q_idx = 0
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        m = QUESTION_LINE_PATTERN.match(line)
        if not m:
            q_idx = 0
            q_text = line
        else:
            q_idx = int(m.group(1))
            q_text = m.group(2)
        options = _extract_options(q_text)
        if not options or len(options) < 2:
            continue
        if q_idx == 0:
            q_idx = max_q_idx + 1
        if q_idx not in questions:
            questions[q_idx] = options
            max_q_idx = max(max_q_idx, q_idx)
    return questions


def _extract_choice(query: str) -> tuple[Optional[int], Optional[str]]:
    q_num_match = QUESTION_NUMBER_PATTERN.search(query)
    option_match = OPTION_LETTER_PATTERN.search(query.upper())
    q_num = int(q_num_match.group(1)) if q_num_match else None
    option = option_match.group(1).upper() if option_match else None
    return q_num, option


def resolve_follow_up_query(query: str, last_turn: Optional[ChatTurn]) -> Optional[FollowUpResolution]:
    raw_query = query.strip()
    if not raw_query:
        return None
    if not SHORT_FOLLOW_UP_PATTERN.match(raw_query):
        return None
    if not last_turn:
        return None

    prev_query = last_turn.query.strip()
    if not prev_query:
        return None

    q_num, option = _extract_choice(raw_query)
    if not q_num and not option:
        return None

    language = _guess_language(prev_query)
    notes = ["follow_up_resolved"]
    questions = _parse_clarification_questions(last_turn.answer)

    if q_num and not option:
        if language == "zh":
            if q_num in questions:
                options_preview = " ".join(
                    [f"{key}) {value}" for key, value in sorted(questions[q_num].items())]
                )
                clarify_message = (
                    f"你已選第{q_num}題，但還沒選選項。"
                    f"請直接回覆第{q_num}題的選項字母（例如 `{q_num}A` 或 `{q_num}B`）。\n"
                    f"第{q_num}題選項：{options_preview}"
                )
            else:
                clarify_message = (
                    f"你已選第{q_num}題，但還沒選選項。"
                    f"請直接回覆選項字母（例如 `{q_num}A` 或 `{q_num}B`）。"
                )
        else:
            if q_num in questions:
                options_preview = " ".join(
                    [f"{key}) {value}" for key, value in sorted(questions[q_num].items())]
                )
                clarify_message = (
                    f"You selected question {q_num}, but no option yet. "
                    f"Reply with the option letter (for example `{q_num}A` or `{q_num}B`).\n"
                    f"Question {q_num} options: {options_preview}"
                )
            else:
                clarify_message = (
                    f"You selected question {q_num}, but no option yet. "
                    f"Reply with an option letter (for example `{q_num}A` or `{q_num}B`)."
                )

        notes.append("follow_up_need_option")
        return FollowUpResolution(
            effective_query=prev_query,
            policy_notes=notes,
            preferred_language=language,
            mode="need_option",
            clarify_message=clarify_message,
        )

    selected_question = q_num
    selected_option_text: Optional[str] = None

    if option and questions:
        if selected_question and selected_question in questions:
            selected_option_text = questions[selected_question].get(option)
        elif not selected_question:
            matched_questions = [q_idx for q_idx in sorted(questions.keys()) if option in questions[q_idx]]
            if len(matched_questions) == 1:
                selected_question = matched_questions[0]
                selected_option_text = questions[selected_question][option]
                notes.append("follow_up_inferred_single_question")
            elif 2 in questions and option in questions[2]:
                selected_question = 2
                selected_option_text = questions[2][option]
                notes.append("follow_up_assumed_question2")
            else:
                for q_idx in sorted(questions.keys()):
                    if option in questions[q_idx]:
                        selected_question = q_idx
                        selected_option_text = questions[q_idx][option]
                        notes.append("follow_up_assumed_question")
                        break

    if language == "zh":
        if option and selected_option_text:
            effective = (
                f"{prev_query}\n"
                f"使用者補充：第{selected_question}題選 {option}（{selected_option_text}）。\n"
                f"原始追問：{raw_query}"
            )
        elif option:
            effective = (
                f"{prev_query}\n"
                f"使用者補充：選項 {option}（題號未明確）。\n"
                f"原始追問：{raw_query}"
            )
            notes.append("follow_up_ambiguous_option")
        else:
            effective = (
                f"{prev_query}\n"
                f"使用者補充：第{selected_question}題。\n"
                f"原始追問：{raw_query}"
            )
            notes.append("follow_up_missing_option")
    else:
        if option and selected_option_text:
            effective = (
                f"{prev_query}\n"
                f"User follow-up: question {selected_question}, option {option} ({selected_option_text}).\n"
                f"Raw follow-up: {raw_query}"
            )
        elif option:
            effective = (
                f"{prev_query}\n"
                f"User follow-up: option {option} (question number not explicit).\n"
                f"Raw follow-up: {raw_query}"
            )
            notes.append("follow_up_ambiguous_option")
        else:
            effective = (
                f"{prev_query}\n"
                f"User follow-up: question {selected_question}.\n"
                f"Raw follow-up: {raw_query}"
            )
            notes.append("follow_up_missing_option")

    return FollowUpResolution(
        effective_query=effective,
        policy_notes=notes,
        preferred_language=language,
    )
