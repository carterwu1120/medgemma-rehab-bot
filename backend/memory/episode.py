from __future__ import annotations

import re
from typing import Any, Optional

from .store import EpisodeState


EXPLICIT_NEW_EPISODE_PATTERN = re.compile(
    r"(另外一個問題|換個問題|不是這個|不是剛剛那個|現在變成|改成|我現在|目前是|new problem|another issue|different problem|now it is)",
    re.IGNORECASE,
)
RESOLUTION_PATTERN = re.compile(
    r"(好了|改善|緩解|不痛了|已經好|解決了|恢復了|resolved|better now|improved|no longer hurts)",
    re.IGNORECASE,
)
MULTI_ISSUE_PATTERN = re.compile(
    r"(兩個都|同時|一起|都會|both|at the same time|simultaneously)",
    re.IGNORECASE,
)
PAIN_SIGNAL_PATTERN = re.compile(r"(痛|痠|酸|刺痛|麻|不舒服|落枕|pain|sore|stiff|numb|wry neck)", re.IGNORECASE)
BODY_BUCKET_PATTERNS = {
    "neck_shoulder": re.compile(r"(頸|脖|肩|斜方|落枕|neck|shoulder|trapezi|wry neck|stiff neck)", re.IGNORECASE),
    "back": re.compile(r"(背|腰|脊椎|下背|back|lumbar|spine)", re.IGNORECASE),
    "arm_hand": re.compile(r"(手腕|前臂|手肘|手|wrist|forearm|elbow|hand)", re.IGNORECASE),
    "leg_foot": re.compile(r"(膝|踝|腳|足|腿|knee|ankle|foot|leg|heel|achilles)", re.IGNORECASE),
}
LATERALITY_PATTERNS = {
    "left": re.compile(r"(左|left)", re.IGNORECASE),
    "right": re.compile(r"(右|right)", re.IGNORECASE),
    "bilateral": re.compile(r"(雙|兩隻|both|bilateral)", re.IGNORECASE),
}
TRIGGER_PATTERNS = {
    "walking": re.compile(r"(走路|步行|walking)", re.IGNORECASE),
    "stairs": re.compile(r"(上下樓|爬樓梯|樓梯|stairs)", re.IGNORECASE),
    "standing": re.compile(r"(久站|站太久|standing)", re.IGNORECASE),
    "running": re.compile(r"(跑步|跑跳|running|jumping)", re.IGNORECASE),
    "sitting": re.compile(r"(久坐|sitting|desk)", re.IGNORECASE),
    "training": re.compile(r"(訓練後|運動後|training|workout|lifting)", re.IGNORECASE),
    "sleeping": re.compile(r"(睡覺|翻身|夜間|sleep|night)", re.IGNORECASE),
    "bending": re.compile(r"(彎腰|bending)", re.IGNORECASE),
    "gripping": re.compile(r"(握拳|拿東西|施力|grip|gripping|carry|carrying)", re.IGNORECASE),
    "typing": re.compile(r"(打字|鍵盤|滑鼠|typing|keyboard|mouse)", re.IGNORECASE),
    "turning_head": re.compile(r"(轉頭|扭頭|turning head)", re.IGNORECASE),
}
PAIN_TYPE_PATTERNS = {
    "stabbing": re.compile(r"(刺痛|刺|stabbing|sharp)", re.IGNORECASE),
    "aching": re.compile(r"(痠痛|酸痛|sore|ache|aching)", re.IGNORECASE),
    "numb": re.compile(r"(麻|麻木|numb|tingling)", re.IGNORECASE),
    "persistent": re.compile(r"(持續性|持續|一直痛|persistent|constant)", re.IGNORECASE),
    "intermittent": re.compile(r"(間歇性|間歇|偶爾痛|intermittent|on and off)", re.IGNORECASE),
}
RED_FLAG_PATTERN = re.compile(
    r"(麻木無力|胸痛|發燒|暈眩|無法行走|numbness|weakness|chest pain|fever|dizziness|cannot walk)",
    re.IGNORECASE,
)
NO_RED_FLAG_PATTERN = re.compile(
    r"(沒有其他症狀|無其他症狀|沒有麻|沒有無力|沒有胸痛|沒有發燒|沒有暈眩|none|no other symptoms|no numbness|no weakness|no chest pain|no fever|no dizziness)",
    re.IGNORECASE,
)
SEVERITY_PATTERN = re.compile(r"\b([0-9]|10)\s*/\s*10\b")
DURATION_PATTERN = re.compile(
    r"\b\d+\s*(h|hr|hrs|hour|hours|day|days|week|weeks|month|months)\b|"
    r"\d+\s*(小時|天|週|周|月)|"
    r"(這幾天|最近|一陣子|幾天|幾週|幾周|幾個月|幾小時|持續性|間歇性|持續|間歇|反覆)",
    re.IGNORECASE,
)


def _detect_body_bucket(text: str) -> Optional[str]:
    for name, pattern in BODY_BUCKET_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def detect_body_bucket(text: str) -> Optional[str]:
    return _detect_body_bucket(text)


def is_resolution_update(query: str) -> bool:
    return bool(RESOLUTION_PATTERN.search(query))


def is_multi_issue_query(query: str) -> bool:
    return bool(MULTI_ISSUE_PATTERN.search(query))


def _extract_slots(text: str) -> dict[str, Any]:
    slots: dict[str, Any] = {}
    body_bucket = _detect_body_bucket(text)
    if body_bucket:
        slots["body_bucket"] = body_bucket

    for key, pattern in LATERALITY_PATTERNS.items():
        if pattern.search(text):
            slots["laterality"] = key
            break

    triggers = [name for name, pattern in TRIGGER_PATTERNS.items() if pattern.search(text)]
    if triggers:
        slots["trigger"] = ",".join(sorted(triggers))

    pain_types = [name for name, pattern in PAIN_TYPE_PATTERNS.items() if pattern.search(text)]
    if pain_types:
        slots["pain_type"] = ",".join(sorted(pain_types))

    duration = DURATION_PATTERN.search(text)
    if duration:
        slots["duration"] = duration.group(0)

    severity = SEVERITY_PATTERN.search(text)
    if severity:
        slots["severity"] = severity.group(0)

    if RED_FLAG_PATTERN.search(text):
        slots["red_flags"] = "yes"
    elif NO_RED_FLAG_PATTERN.search(text):
        slots["red_flags"] = "no"

    return slots


def should_start_new_episode(query: str, active_episode: Optional[EpisodeState]) -> bool:
    if not active_episode:
        return True
    if EXPLICIT_NEW_EPISODE_PATTERN.search(query):
        return True

    active_body = str(active_episode.slots.get("body_bucket", "")).strip()
    next_body = _detect_body_bucket(query)
    if active_body and next_body and active_body != next_body and PAIN_SIGNAL_PATTERN.search(query):
        return True

    return False


def extract_slot_updates(query: str, *, previous_slots: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    updates = _extract_slots(query)
    if not previous_slots:
        return updates
    filtered: dict[str, Any] = {}
    for key, value in updates.items():
        if previous_slots.get(key) != value:
            filtered[key] = value
    return filtered


def build_episode_context(active_episode: Optional[EpisodeState]) -> Optional[str]:
    if not active_episode:
        return None
    if not active_episode.slots:
        return None
    pairs = [f"{key}={value}" for key, value in sorted(active_episode.slots.items()) if str(value).strip()]
    if not pairs:
        return None
    return "Current problem slots: " + "; ".join(pairs)


def has_minimum_slots_for_plan(slots: Optional[dict[str, Any]]) -> bool:
    if not slots:
        return False
    body_ok = bool(slots.get("body_bucket"))
    trigger_ok = bool(slots.get("trigger"))
    detail_ok = bool(slots.get("duration") or slots.get("severity") or slots.get("pain_type"))
    # Avoid endless clarification loops: body+trigger is enough for an initial plan.
    return body_ok and (trigger_ok or detail_ok)
