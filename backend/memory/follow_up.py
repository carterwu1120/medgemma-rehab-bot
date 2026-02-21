from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

from .store import ChatTurn


QUESTION_LINE_PATTERN = re.compile(r"^\s*([1-3])[)\.]?\s*(.+?)\s*$")
OPTION_MARKER_PATTERN = re.compile(r"([A-Da-d])\s*[)）.:：-]")
QUESTION_NUMBER_PATTERN = re.compile(r"(?:^|\b|第)\s*([1-3])\s*(?:題|question)?", re.IGNORECASE)
OPTION_LETTER_PATTERN = re.compile(r"\b([A-Da-d])\b")
EXPLICIT_PAIR_PATTERN = re.compile(r"\b([1-3])\s*[-:：)]?\s*([A-Da-d])\b")
SHORT_FOLLOW_UP_PATTERN = re.compile(
    r"^\s*(?:如果是|if|選|option|答案|answer|第?\s*[1-3]\s*題?)?\s*[1-3]?\s*[A-Da-d]?\s*$",
    re.IGNORECASE,
)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
PURE_OPTION_LINE_PATTERN = re.compile(r"^\s*(?:第?\s*[1-3]\s*題?\s*)?[A-Da-d]\s*$", re.IGNORECASE)
PURE_QNUM_LINE_PATTERN = re.compile(r"^\s*[1-3]\s*[)\.]?\s*$")
TRIGGER_HINT_PATTERN = re.compile(
    r"(走路|走動|久坐|久站|抬手|轉頭|彎腰|訓練後|運動後|睡覺|翻身|夜間|walking|running|sitting|standing|lifting|sleep)",
    re.IGNORECASE,
)
NEW_REQUEST_PATTERN = re.compile(
    r"(請|可以|怎麼|如何|給我|建議|計畫|方案|should|what|how|can you|plan|advice)",
    re.IGNORECASE,
)
PAIN_SIGNAL_PATTERN = re.compile(r"(痛|痠|酸|僵硬|麻|不舒服|pain|sore|stiff|numb)", re.IGNORECASE)
BODY_BUCKET_PATTERNS = {
    "neck_shoulder": re.compile(r"(頸|肩|斜方|neck|shoulder|trapezi)", re.IGNORECASE),
    "back": re.compile(r"(背|腰|脊椎|下背|back|lumbar|spine)", re.IGNORECASE),
    "arm_hand": re.compile(r"(手腕|前臂|手肘|手|wrist|forearm|elbow|hand)", re.IGNORECASE),
    "leg_foot": re.compile(r"(膝|踝|腳|足|腿|knee|ankle|foot|leg|heel|achilles)", re.IGNORECASE),
}


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


def _extract_explicit_pair(query: str) -> tuple[Optional[int], Optional[str]]:
    pair_match = EXPLICIT_PAIR_PATTERN.search(query)
    if not pair_match:
        return None, None
    return int(pair_match.group(1)), pair_match.group(2).upper()


def _extract_free_text_lines(query: str) -> list[str]:
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    free: list[str] = []
    for line in lines:
        if PURE_OPTION_LINE_PATTERN.match(line) or PURE_QNUM_LINE_PATTERN.match(line):
            continue
        if EXPLICIT_PAIR_PATTERN.search(line):
            continue
        free.append(line)
    return free


def _semantic_match_from_text(
    text_lines: list[str],
    questions: dict[int, dict[str, str]],
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    if not text_lines or not questions:
        return None, None, None

    matches: list[tuple[int, str, str]] = []
    for snippet in text_lines:
        compact = snippet.strip().lower()
        if len(compact) < 2:
            continue
        for q_idx, options in questions.items():
            for letter, option_text in options.items():
                opt = option_text.lower()
                if compact in opt or opt in compact:
                    matches.append((q_idx, letter, option_text))

    if not matches:
        return None, None, None

    # Prefer unique question/option hit.
    uniq = list(dict.fromkeys(matches))
    if len(uniq) == 1:
        return uniq[0]

    # If multiple hits, prefer trigger-like text for question-2 style options.
    if any(TRIGGER_HINT_PATTERN.search(x) for x in text_lines):
        for q_idx, letter, option_text in uniq:
            if q_idx == 2:
                return q_idx, letter, option_text

    return uniq[0]


def _is_candidate_follow_up(raw_query: str, has_questions: bool) -> bool:
    if not has_questions:
        return False
    if SHORT_FOLLOW_UP_PATTERN.match(raw_query):
        return True
    if EXPLICIT_PAIR_PATTERN.search(raw_query):
        return True
    if OPTION_LETTER_PATTERN.search(raw_query) and len(raw_query) <= 64:
        return True
    return False


def _is_clarification_turn(last_turn: ChatTurn) -> bool:
    notes = set(last_turn.policy_notes or [])
    if "clarification_mode" in notes or "clarify_first_only" in notes:
        return True
    answer_lower = last_turn.answer.lower()
    return "先確認" in last_turn.answer or "one quick check" in answer_lower or "please confirm" in answer_lower


def _looks_like_freeform_follow_up(raw_query: str) -> bool:
    compact = " ".join(raw_query.strip().split())
    if not compact:
        return False
    if len(compact) > 120:
        return False
    if NEW_REQUEST_PATTERN.search(compact):
        return False
    return True


def _expand_freeform_follow_up(*, prev_query: str, raw_query: str, language: str) -> str:
    if language == "zh":
        return f"{prev_query}\n使用者補充資訊：{raw_query}"
    return f"{prev_query}\nUser extra detail: {raw_query}"


def _infer_body_buckets(text: str) -> set[str]:
    buckets: set[str] = set()
    for name, pattern in BODY_BUCKET_PATTERNS.items():
        if pattern.search(text):
            buckets.add(name)
    return buckets


def _looks_like_new_primary_query(raw_query: str) -> bool:
    has_body = bool(_infer_body_buckets(raw_query))
    has_pain = bool(PAIN_SIGNAL_PATTERN.search(raw_query))
    return has_body and has_pain


def _is_topic_shift(prev_query: str, raw_query: str) -> bool:
    prev_buckets = _infer_body_buckets(prev_query)
    raw_buckets = _infer_body_buckets(raw_query)
    if not prev_buckets or not raw_buckets:
        return False
    return prev_buckets.isdisjoint(raw_buckets)


def resolve_follow_up_query(query: str, last_turn: Optional[ChatTurn]) -> Optional[FollowUpResolution]:
    raw_query = query.strip()
    if not raw_query:
        return None
    if not last_turn:
        return None

    prev_query = last_turn.query.strip()
    if not prev_query:
        return None

    language = _guess_language(f"{prev_query}\n{raw_query}")
    clarify_turn = _is_clarification_turn(last_turn)
    questions = _parse_clarification_questions(last_turn.answer)
    if not questions:
        if clarify_turn and _looks_like_freeform_follow_up(raw_query):
            if _looks_like_new_primary_query(raw_query) and _is_topic_shift(prev_query, raw_query):
                return None
            return FollowUpResolution(
                effective_query=_expand_freeform_follow_up(prev_query=prev_query, raw_query=raw_query, language=language),
                policy_notes=["follow_up_resolved", "follow_up_freeform_clarification"],
                preferred_language=language,
            )
        return None

    if not _is_candidate_follow_up(raw_query, has_questions=True):
        if clarify_turn and _looks_like_freeform_follow_up(raw_query):
            if _looks_like_new_primary_query(raw_query) and _is_topic_shift(prev_query, raw_query):
                return None
            return FollowUpResolution(
                effective_query=_expand_freeform_follow_up(prev_query=prev_query, raw_query=raw_query, language=language),
                policy_notes=["follow_up_resolved", "follow_up_freeform_with_options"],
                preferred_language=language,
            )
        return None

    explicit_q_num, explicit_option = _extract_explicit_pair(raw_query)
    q_num, option = _extract_choice(raw_query)
    if explicit_q_num:
        q_num = explicit_q_num
    if explicit_option:
        option = explicit_option

    free_text_lines = _extract_free_text_lines(raw_query)
    semantic_q, semantic_option, semantic_text = _semantic_match_from_text(free_text_lines, questions)
    if not q_num and not option:
        if clarify_turn and _looks_like_freeform_follow_up(raw_query):
            if _looks_like_new_primary_query(raw_query) and _is_topic_shift(prev_query, raw_query):
                return None
            return FollowUpResolution(
                effective_query=_expand_freeform_follow_up(prev_query=prev_query, raw_query=raw_query, language=language),
                policy_notes=["follow_up_resolved", "follow_up_freeform_no_option"],
                preferred_language=language,
            )
        return None

    notes = ["follow_up_resolved"]

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
    selected_option = option
    selected_option_text: Optional[str] = None

    if semantic_q and semantic_option and semantic_text:
        if not selected_question and not selected_option:
            selected_question = semantic_q
            selected_option = semantic_option
            selected_option_text = semantic_text
            notes.append("follow_up_semantic_match")
        elif not selected_question and selected_option and selected_option != semantic_option:
            selected_question = semantic_q
            selected_option = semantic_option
            selected_option_text = semantic_text
            notes.append("follow_up_option_conflict_used_semantic")

    if selected_option and questions:
        if selected_question and selected_question in questions:
            if selected_option_text is None:
                selected_option_text = questions[selected_question].get(selected_option)
        elif not selected_question:
            matched_questions = [q_idx for q_idx in sorted(questions.keys()) if selected_option in questions[q_idx]]
            if len(matched_questions) == 1:
                selected_question = matched_questions[0]
                selected_option_text = questions[selected_question][selected_option]
                notes.append("follow_up_inferred_single_question")
            elif 2 in questions and selected_option in questions[2]:
                selected_question = 2
                selected_option_text = questions[2][selected_option]
                notes.append("follow_up_assumed_question2")
            else:
                for q_idx in sorted(questions.keys()):
                    if selected_option in questions[q_idx]:
                        selected_question = q_idx
                        selected_option_text = questions[q_idx][selected_option]
                        notes.append("follow_up_assumed_question")
                        break

    free_text_note = ""
    if free_text_lines:
        merged = "；".join(free_text_lines[:2])
        if language == "zh":
            free_text_note = f"使用者額外描述：{merged}。\n"
        else:
            free_text_note = f"User extra detail: {merged}.\n"
        notes.append("follow_up_with_free_text")

    if language == "zh":
        if selected_option and selected_option_text:
            effective = (
                f"{prev_query}\n"
                f"使用者補充：第{selected_question}題選 {selected_option}（{selected_option_text}）。\n"
                f"{free_text_note}"
                f"原始追問：{raw_query}"
            )
        elif selected_option:
            effective = (
                f"{prev_query}\n"
                f"使用者補充：選項 {selected_option}（題號未明確）。\n"
                f"{free_text_note}"
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
        if selected_option and selected_option_text:
            effective = (
                f"{prev_query}\n"
                f"User follow-up: question {selected_question}, option {selected_option} ({selected_option_text}).\n"
                f"{free_text_note}"
                f"Raw follow-up: {raw_query}"
            )
        elif selected_option:
            effective = (
                f"{prev_query}\n"
                f"User follow-up: option {selected_option} (question number not explicit).\n"
                f"{free_text_note}"
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
