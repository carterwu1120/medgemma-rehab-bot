import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from backend.llm_api import VLLMApi

from .bm25 import BM25Retriever, RetrievedChunk
from .query_rewrite import QueryRewriter, QueryVariant, default_query_rewriter


SYSTEM_PROMPT = (
    "You are a cautious home rehabilitation assistant. "
    "Do not diagnose. "
    "Always answer in the same language as the user's query. "
    "Only use retrieved evidence; do not invent treatment details."
)

SAFETY_TAGS = {"safety_red_flags", "safety_stop_rules"}
SAFETY_ROUTE_SUFFIX = " stop exercise seek medical care red flag 停止運動 就醫 紅旗 立即停止"
SAFETY_QUERY_PATTERNS = [
    re.compile(
        r"\b(stop|urgent|emergency|seek medical care|seek care|see a doctor|go to er|red flag|dizziness|chest pain|numbness|weakness|fever|night pain|should i continue|can i keep training|can i continue)\b",
        re.IGNORECASE,
    ),
    re.compile(r"(停止|就醫|急診|紅旗|胸痛|暈|麻木|無力|發燒|夜間痛|惡化|繼續練|繼續運動|要不要停|是否停止|可不可以繼續|還要繼續嗎|還能繼續嗎|繼續嗎)"),
]
BODY_HINT_PATTERNS: Dict[str, re.Pattern[str]] = {
    "body_shoulder": re.compile(r"\b(shoulder|rotator cuff|scapula)\b|肩|肩膀|旋轉肌袖", re.IGNORECASE),
    "body_neck_trap": re.compile(r"\b(neck|cervical|trapezius)\b|頸|肩頸|斜方肌", re.IGNORECASE),
    "body_back_spine": re.compile(r"\b(back|lumbar|thoracic|spine|low back)\b|背|腰|脊椎|下背", re.IGNORECASE),
    "body_knee": re.compile(r"\b(knee|patella|meniscus)\b|膝|膝蓋|半月板", re.IGNORECASE),
    "body_ankle_foot": re.compile(
        r"\b(ankle|foot|heel|achilles|plantar)\b|踝|腳踝|腳跟|足底|足弓|跟腱|腳",
        re.IGNORECASE,
    ),
    "body_hip_glute": re.compile(r"\b(hip|glute|buttock)\b|髖|臀|臀肌", re.IGNORECASE),
    "body_elbow_wrist_hand": re.compile(r"\b(elbow|wrist|forearm|hand)\b|手肘|手腕|前臂|手", re.IGNORECASE),
}
CHUNK_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+_p\d+_c\d+")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
URL_PATTERN = re.compile(r"https?://\S+")
MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")
DURATION_PATTERN = re.compile(
    r"\b\d+\s*(h|hr|hrs|hour|hours|day|days|week|weeks|month|months)\b|"
    r"\d+\s*(小時|天|週|周|月)|"
    r"(這幾天|最近|一陣子|幾天|幾週|幾周|幾個月|幾小時|持續性|間歇性|持續|間歇|反覆)",
    re.IGNORECASE,
)
SEVERITY_PATTERN = re.compile(
    r"\b([0-9]|10)\s*/\s*10\b|"
    r"\b(pain|sore|stiff|numb|tingling|weakness)\b|"
    r"(痛|痠|酸|僵硬|緊|麻|刺痛|無力)",
    re.IGNORECASE,
)
TRIGGER_PATTERN = re.compile(
    r"\b(after|during|when|post|training|lifting|running|sitting|sleep|desk)\b|"
    r"(運動後|訓練後|久坐|久站|走路|步行|睡前|睡覺|夜間|翻身|起床|工作後|抬手|彎腰|跑步後|持續性|間歇性|持續|間歇)",
    re.IGNORECASE,
)
RED_FLAG_PATTERN = re.compile(
    r"\b(numbness|weakness|chest pain|dizziness|fever|night pain|cannot walk|unable to walk)\b|"
    r"(麻木|無力|胸痛|暈眩|發燒|夜間痛|無法行走|走不了)",
    re.IGNORECASE,
)
ALT_ASK_PATTERN = re.compile(
    r"(除了|另外|還有|其他|替代|alternative|else|another option|other options|anything else)",
    re.IGNORECASE,
)


@dataclass
class RAGResult:
    answer: str
    retrieved: List[RetrievedChunk]
    context_text: str
    policy_notes: List[str]


class RehabRAG:
    def __init__(
        self,
        retriever: BM25Retriever,
        llm_api: Optional[VLLMApi] = None,
        system_prompt: str = SYSTEM_PROMPT,
        candidate_pool: int = 80,
        safety_boost: float = 0.08,
        safety_route: bool = True,
        body_boost: float = 0.35,
        body_mismatch_penalty: float = 0.20,
        body_min_hits: int = 2,
        query_rewriter: Optional[QueryRewriter] = None,
        rrf_k: int = 60,
    ) -> None:
        self.retriever = retriever
        self.llm_api = llm_api
        self.system_prompt = system_prompt
        self.candidate_pool = max(5, candidate_pool)
        self.safety_boost = max(0.0, safety_boost)
        self.safety_route = safety_route
        self.body_boost = max(0.0, body_boost)
        self.body_mismatch_penalty = max(0.0, body_mismatch_penalty)
        self.body_min_hits = max(0, body_min_hits)
        self.query_rewriter = query_rewriter if query_rewriter is not None else default_query_rewriter()
        self.rrf_k = max(1, rrf_k)
        self.clarify_first_only = os.getenv("RAG_CLARIFY_FIRST_ONLY", "0") == "1"
        self.clarification_mode = os.getenv("RAG_CLARIFICATION_MODE", "hybrid").strip().lower()
        self.response_style = os.getenv("RAG_RESPONSE_STYLE", "natural").strip().lower()
        self.retry_on_short = os.getenv("RAG_RETRY_SHORT_ANSWER", "1") == "1"
        self.diversity_enabled = os.getenv("RAG_DIVERSITY_ENABLED", "1") == "1"
        self.diversity_lambda = min(0.95, max(0.05, float(os.getenv("RAG_DIVERSITY_LAMBDA", "0.75"))))
        self.slot_query_expansion = os.getenv("RAG_SLOT_QUERY_EXPANSION", "1") == "1"
        self.strict_safety_fallback = os.getenv("RAG_STRICT_SAFETY_FALLBACK", "0") == "1"
        self.generation_top_p = min(1.0, max(0.1, float(os.getenv("RAG_GEN_TOP_P", "0.9"))))
        self.generation_repetition_penalty = max(1.0, float(os.getenv("RAG_GEN_REPETITION_PENALTY", "1.08")))

    def _gen_extra(self) -> Dict[str, float]:
        return {
            "top_p": self.generation_top_p,
            "repetition_penalty": self.generation_repetition_penalty,
        }

    def _retrieve_candidates(self, query: str, top_k: int) -> tuple[List[RetrievedChunk], List[str]]:
        notes: List[str] = []
        if not self.query_rewriter:
            return self.retriever.search(query=query, top_k=top_k), notes

        variants = self.query_rewriter.rewrite(query)
        if not variants:
            variants = [QueryVariant(text=query, source="original", weight=1.0)]

        if len(variants) > 1:
            notes.append(f"query_rewrite_variants={len(variants)}")
            notes.append("query_rewrite_sources=" + ",".join(sorted({v.source for v in variants if v.source != "original"})))

        variant_hits: List[tuple[QueryVariant, List[RetrievedChunk]]] = []
        for variant in variants:
            hits = self.retriever.search(query=variant.text, top_k=top_k)
            if hits:
                variant_hits.append((variant, hits))

        if not variant_hits:
            return [], notes

        # Reciprocal-rank fusion keeps this extensible across heterogeneous retrievers.
        fused_scores: Dict[str, float] = {}
        chunk_by_id: Dict[str, RetrievedChunk] = {}
        for variant, hits in variant_hits:
            for rank, chunk in enumerate(hits, start=1):
                fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                    float(variant.weight) / float(self.rrf_k + rank)
                )
                prev = chunk_by_id.get(chunk.chunk_id)
                if prev is None or chunk.score > prev.score:
                    chunk_by_id[chunk.chunk_id] = chunk

        ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)
        fused: List[RetrievedChunk] = []
        for chunk_id, score in ordered[:top_k]:
            base = chunk_by_id[chunk_id]
            fused.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=base.chunk_id,
                    text=base.text,
                    source_name=base.source_name,
                    page=base.page,
                    title=base.title,
                    tags=base.tags,
                )
            )

        return fused, notes

    @staticmethod
    def _has_safety_intent(query: str) -> bool:
        return any(pattern.search(query) for pattern in SAFETY_QUERY_PATTERNS)

    @staticmethod
    def _has_safety_tags(chunks: List[RetrievedChunk]) -> bool:
        return any(bool(SAFETY_TAGS & set(chunk.tags)) for chunk in chunks)

    @staticmethod
    def _infer_expected_body_tags(query: str) -> set[str]:
        expected: set[str] = set()
        for tag, pattern in BODY_HINT_PATTERNS.items():
            if pattern.search(query):
                expected.add(tag)
        return expected

    @staticmethod
    def _detect_query_language(query: str) -> str:
        # Minimal language routing for current corpus/use-cases.
        return "zh" if CJK_PATTERN.search(query) else "en"

    @staticmethod
    def _extract_query_slots(query: str) -> Dict[str, bool]:
        has_body = bool(RehabRAG._infer_expected_body_tags(query))
        has_duration = bool(DURATION_PATTERN.search(query))
        has_severity = bool(SEVERITY_PATTERN.search(query))
        has_trigger = bool(TRIGGER_PATTERN.search(query))
        has_red_flags = bool(RED_FLAG_PATTERN.search(query))
        return {
            "has_body": has_body,
            "has_duration": has_duration,
            "has_severity": has_severity,
            "has_trigger": has_trigger,
            "has_red_flags": has_red_flags,
        }

    @staticmethod
    def _is_query_vague(query: str) -> bool:
        normalized = " ".join(query.strip().split())
        if len(normalized) <= 6:
            return True
        slots = RehabRAG._extract_query_slots(query)
        detail_score = sum([slots["has_duration"], slots["has_severity"], slots["has_trigger"], slots["has_red_flags"]])

        # Body-only or body+single-detail often needs clarification before prescribing steps.
        if slots["has_body"]:
            if slots["has_red_flags"] and slots["has_severity"]:
                return False
            if slots["has_trigger"] and (slots["has_duration"] or slots["has_severity"]):
                return False
            if slots["has_duration"] and slots["has_severity"]:
                return False
            return True

        return detail_score <= 1

    def _apply_body_policy(
        self,
        query: str,
        hits: List[RetrievedChunk],
        top_k: int,
    ) -> tuple[List[RetrievedChunk], List[str]]:
        if not hits:
            return [], []

        expected = self._infer_expected_body_tags(query)
        if not expected:
            return list(hits), []

        notes: List[str] = []
        rescored: List[RetrievedChunk] = []
        for chunk in hits:
            tags = set(chunk.tags)
            has_expected = bool(expected & tags)
            has_other_body = any(t.startswith("body_") for t in tags) and not has_expected

            score = float(chunk.score)
            if has_expected:
                score *= (1.0 + self.body_boost)
            elif has_other_body:
                score *= max(0.0, 1.0 - self.body_mismatch_penalty)

            rescored.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_name=chunk.source_name,
                    page=chunk.page,
                    title=chunk.title,
                    tags=chunk.tags,
                )
            )

        ranked = sorted(rescored, key=lambda x: x.score, reverse=True)
        notes.append("applied_body_boost")

        if self.body_min_hits > 0:
            head = ranked[:top_k]
            head_expected = [c for c in head if expected & set(c.tags)]
            if len(head_expected) < self.body_min_hits:
                need = self.body_min_hits - len(head_expected)
                tail_expected = [c for c in ranked[top_k:] if expected & set(c.tags)]
                if tail_expected:
                    inject = tail_expected[:need]
                    keep = [c for c in head if not (expected & set(c.tags))]
                    merged = head_expected + inject + keep
                    reordered_ids = {c.chunk_id for c in merged[:top_k]}
                    rest = [c for c in ranked if c.chunk_id not in reordered_ids]
                    ranked = merged[:top_k] + rest
                    notes.append("enforced_body_coverage")

        return ranked, notes

    def _apply_safety_policy(self, query: str, hits: List[RetrievedChunk], top_k: int) -> tuple[List[RetrievedChunk], List[str]]:
        if not hits:
            return [], []

        notes: List[str] = []
        safety_intent = self._has_safety_intent(query)
        ranked = list(hits)

        if safety_intent and self.safety_boost > 0:
            rescored: List[RetrievedChunk] = []
            for chunk in ranked:
                boost = self.safety_boost if (SAFETY_TAGS & set(chunk.tags)) else 0.0
                rescored.append(
                    RetrievedChunk(
                        score=float(chunk.score) + boost,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source_name=chunk.source_name,
                        page=chunk.page,
                        title=chunk.title,
                        tags=chunk.tags,
                    )
                )
            ranked = sorted(rescored, key=lambda x: x.score, reverse=True)
            notes.append("applied_safety_boost")

        top = ranked[:top_k]
        if safety_intent and self.safety_route and not self._has_safety_tags(top):
            expanded = f"{query}{SAFETY_ROUTE_SUFFIX}"
            fallback_hits = self.retriever.search(expanded, top_k=max(self.candidate_pool, top_k, 20))
            fallback_safety = [chunk for chunk in fallback_hits if SAFETY_TAGS & set(chunk.tags)]
            if fallback_safety:
                expected_body = self._infer_expected_body_tags(query)
                if expected_body:
                    body_aligned = [c for c in fallback_safety if expected_body & set(c.tags)]
                    best_safety = body_aligned[0] if body_aligned else fallback_safety[0]
                else:
                    best_safety = fallback_safety[0]

                routed = [best_safety]
                routed.extend([chunk for chunk in top if chunk.chunk_id != best_safety.chunk_id])
                top = routed[:top_k]
                notes.append("routed_safety_chunk")

        return top, notes

    @staticmethod
    def _is_alternative_request(query: str) -> bool:
        return bool(ALT_ASK_PATTERN.search(query))

    @staticmethod
    def _keyword_overlap(a: RetrievedChunk, b: RetrievedChunk) -> float:
        ta = set(BM25Retriever.tokenize(f"{a.title} {a.text}"))  # type: ignore[arg-type]
        tb = set(BM25Retriever.tokenize(f"{b.title} {b.text}"))  # type: ignore[arg-type]
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        denom = max(1, min(len(ta), len(tb)))
        return inter / float(denom)

    def _apply_diversity(self, hits: List[RetrievedChunk], top_k: int) -> tuple[List[RetrievedChunk], List[str]]:
        if not hits:
            return [], []
        if not self.diversity_enabled:
            return hits[:top_k], []

        selected: List[RetrievedChunk] = []
        pool = list(hits)
        notes: List[str] = []

        while pool and len(selected) < top_k:
            best_idx = 0
            best_score = None
            for idx, cand in enumerate(pool):
                relevance = float(cand.score)
                if not selected:
                    mmr = relevance
                else:
                    redundancy = max(self._keyword_overlap(cand, s) for s in selected)
                    mmr = self.diversity_lambda * relevance - (1.0 - self.diversity_lambda) * redundancy
                if best_score is None or mmr > best_score:
                    best_score = mmr
                    best_idx = idx
            selected.append(pool.pop(best_idx))

        if selected and len(selected) == top_k:
            notes.append("applied_diversity_mmr")
        return selected, notes

    @staticmethod
    def _slot_to_query_terms(known_slots: Optional[Dict[str, str]]) -> str:
        if not known_slots:
            return ""
        body = known_slots.get("body_bucket", "")
        trigger = known_slots.get("trigger", "")
        pain_type = known_slots.get("pain_type", "")
        duration = known_slots.get("duration", "")

        body_terms = {
            "arm_hand": "手 手腕 前臂 手肘 hand wrist forearm elbow grip",
            "leg_foot": "腳 踝 膝 足 小腿 foot ankle knee leg heel achilles",
            "back": "下背 腰 背 脊椎 low back lumbar spine",
            "neck_shoulder": "頸 肩 斜方肌 neck shoulder trapezius",
        }

        parts: List[str] = []
        if body in body_terms:
            parts.append(body_terms[body])
        if trigger:
            parts.append(trigger.replace(",", " "))
        if pain_type:
            parts.append(pain_type.replace(",", " "))
        if duration:
            parts.append(duration)
        return " ".join(parts).strip()

    def _augment_query_with_slots(self, query: str, known_slots: Optional[Dict[str, str]]) -> str:
        if not self.slot_query_expansion:
            return query
        slot_terms = self._slot_to_query_terms(known_slots)
        if not slot_terms:
            return query
        return f"{query}\n{slot_terms}"

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        retrieval_k = max(top_k, self.candidate_pool)
        base_hits, _ = self._retrieve_candidates(query=query, top_k=retrieval_k)
        body_hits, _ = self._apply_body_policy(query=query, hits=base_hits, top_k=top_k)
        hits, _ = self._apply_safety_policy(query=query, hits=body_hits, top_k=top_k)
        return hits

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks, start=1):
            tags = ", ".join(c.tags) if c.tags else "none"
            lines.append(
                f"[{i}] source={c.source_name} page={c.page} chunk_id={c.chunk_id} tags={tags}\n{c.text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def _extract_chunk_refs(text: str) -> List[str]:
        return sorted(set(CHUNK_ID_PATTERN.findall(text)))

    @staticmethod
    def _ensure_reference_block(answer: str, chunks: List[RetrievedChunk], max_refs: int = 3) -> str:
        if not chunks:
            return answer

        refs_in_answer = set(RehabRAG._extract_chunk_refs(answer))
        retrieved_ids = [chunk.chunk_id for chunk in chunks]
        valid_refs = [chunk_id for chunk_id in retrieved_ids if chunk_id in refs_in_answer]
        if valid_refs:
            return answer

        appended = "\n\nReferences:\n"
        for chunk_id in retrieved_ids[: max(1, max_refs)]:
            appended += f"- {chunk_id}\n"
        return answer.rstrip() + appended

    @staticmethod
    def _slots_with_known(slots: Dict[str, bool], known_slots: Optional[Dict[str, str]]) -> Dict[str, bool]:
        merged = dict(slots)
        if not known_slots:
            return merged
        if known_slots.get("body_bucket"):
            merged["has_body"] = True
        if known_slots.get("trigger"):
            merged["has_trigger"] = True
        if known_slots.get("duration"):
            merged["has_duration"] = True
        if known_slots.get("severity") or known_slots.get("pain_type"):
            merged["has_severity"] = True
        if known_slots.get("red_flags"):
            merged["has_red_flags"] = True
        return merged

    @staticmethod
    def _body_bucket_hint(query: str, known_slots: Optional[Dict[str, str]]) -> str:
        if known_slots and known_slots.get("body_bucket"):
            return str(known_slots["body_bucket"])
        expected = RehabRAG._infer_expected_body_tags(query)
        if "body_elbow_wrist_hand" in expected:
            return "arm_hand"
        if "body_ankle_foot" in expected or "body_knee" in expected or "body_hip_glute" in expected:
            return "leg_foot"
        if "body_back_spine" in expected:
            return "back"
        if "body_neck_trap" in expected or "body_shoulder" in expected:
            return "neck_shoulder"
        return "generic"

    @staticmethod
    def _recently_asked_slots(conversation_context: Optional[str]) -> set[str]:
        if not conversation_context:
            return set()
        text = conversation_context.lower()
        asked: set[str] = set()
        patterns = {
            "body_location": ["位置", "部位", "where is", "location"],
            "trigger_situation": ["情境", "誘發", "什麼動作", "what triggers", "trigger"],
            "duration": ["多久", "how long", "duration", "持續"],
            "severity": ["幾分", "0-10", "pain level"],
            "red_flag_check": ["麻木無力", "紅旗", "red flag", "numbness", "chest pain", "dizziness"],
        }
        for slot_name, keys in patterns.items():
            if any(key in text for key in keys):
                asked.add(slot_name)
        return asked

    def _build_clarification_block(
        self,
        query: str,
        language: str,
        known_slots: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        slots = self._slots_with_known(self._extract_query_slots(query), known_slots)
        missing = self._missing_slots_for_prompt(slots)
        asked_recently = self._recently_asked_slots(conversation_context)
        target = next((m for m in missing if m not in asked_recently), missing[0] if missing else "trigger_situation")
        body_bucket = self._body_bucket_hint(query, known_slots)

        if language == "zh":
            question_map = {
                "body_location": "主要不舒服的位置在哪裡？（例如：右腳踝外側、下背中央、左肩前側）",
                "trigger_situation": (
                    "哪一個情境最容易誘發疼痛？請用你的實際活動描述。"
                    if body_bucket == "generic"
                    else (
                        "請描述手/前臂最容易痛的情境（例如拿東西、握拳、打字、扭毛巾）。"
                        if body_bucket == "arm_hand"
                        else (
                            "請描述腳/腿最容易痛的情境（例如走路、上下樓、久站、跑跳後）。"
                            if body_bucket == "leg_foot"
                            else (
                                "請描述背部最容易痛的情境（例如久坐、彎腰、起身、搬東西後）。"
                                if body_bucket == "back"
                                else "請描述頸肩最容易痛的情境（例如抬手、轉頭、久坐、睡醒後）。"
                            )
                        )
                    )
                ),
                "duration": "這個不適大概持續多久了？",
                "severity": "目前疼痛大約幾分（0-10），以及會不會影響日常動作？",
                "red_flag_check": "有沒有麻木無力、胸痛、發燒、暈眩或無法行走？",
            }
            question = question_map.get(target, question_map["trigger_situation"])
            return (
                f"我先確認一件事：{question}\n"
                "若出現麻木無力、胸痛、暈眩、發燒或無法行走，請立即停止並就醫。"
            )

        question_map_en = {
            "body_location": "Where is the main discomfort exactly (for example: right outer ankle, central low back, left front shoulder)?",
            "trigger_situation": (
                "What movement most reliably triggers pain?"
                if body_bucket == "generic"
                else (
                    "What triggers it most: lifting/gripping, typing, or twisting motions?"
                    if body_bucket == "arm_hand"
                    else (
                        "What triggers it most: walking, stairs, prolonged standing, or after running?"
                        if body_bucket == "leg_foot"
                        else (
                            "What triggers it most: long sitting, bending, standing up, or after lifting?"
                            if body_bucket == "back"
                            else "What triggers it most: turning head, lifting arm, long sitting, or after sleep?"
                        )
                    )
                )
            ),
            "duration": "How long has this been going on?",
            "severity": "What is the pain level now (0-10), and does it limit daily movement?",
            "red_flag_check": "Any numbness/weakness, chest pain, fever, dizziness, or inability to walk?",
        }
        question = question_map_en.get(target, question_map_en["trigger_situation"])
        return (
            f"One quick check before I give a precise plan: {question}\n"
            "Please answer in one sentence (no A/B format needed).\n"
                "If numbness/weakness, chest pain, dizziness, fever, or inability to walk appears, stop and seek urgent care."
        )

    def _ensure_clarification_block(
        self,
        answer: str,
        query: str,
        language: str,
        known_slots: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        lowered = answer.lower()
        has_block = (
            "我先確認一件事" in answer
            or "先確認" in answer
            or "one quick check" in lowered
            or "before i give a precise plan" in lowered
        )
        if has_block:
            return answer
        block = self._build_clarification_block(
            query,
            language=language,
            known_slots=known_slots,
            conversation_context=conversation_context,
        )
        return f"{block}\n{answer}".strip()

    def _build_clarify_first_response(
        self,
        query: str,
        language: str,
        known_slots: Optional[Dict[str, str]] = None,
        conversation_context: Optional[str] = None,
    ) -> str:
        return self._build_clarification_block(
            query=query,
            language=language,
            known_slots=known_slots,
            conversation_context=conversation_context,
        )

    @staticmethod
    def _missing_slots_for_prompt(slots: Dict[str, bool]) -> List[str]:
        labels = [
            ("has_body", "body_location"),
            ("has_trigger", "trigger_situation"),
            ("has_duration", "duration"),
            ("has_severity", "severity"),
            ("has_red_flags", "red_flag_check"),
        ]
        return [name for key, name in labels if not slots.get(key, False)]

    def _build_dynamic_clarify_prompt(
        self,
        *,
        query: str,
        language: str,
        slots: Dict[str, bool],
        conversation_context: Optional[str],
        known_slots: Optional[Dict[str, str]] = None,
    ) -> str:
        slots = self._slots_with_known(slots, known_slots)
        missing = ", ".join(self._missing_slots_for_prompt(slots)) or "none"
        history = conversation_context if conversation_context else ("（無）" if language == "zh" else "(none)")
        if language == "zh":
            return (
                "你現在在「釐清模式」，不要提供治療計畫。\n"
                "任務：根據使用者原句與上下文，提出 1 題最關鍵追問，不要重問已知資訊。\n"
                "先檢查歷史上下文與 Known slots；已經提供過的欄位禁止再問。\n"
                "如果看起來部位切換成新問題，先一句確認是否要切換主題，並詢問舊問題是否已緩解。\n"
                "已知欄位（true 代表已提供）：\n"
                f"- body_location={slots.get('has_body', False)}\n"
                f"- trigger_situation={slots.get('has_trigger', False)}\n"
                f"- duration={slots.get('has_duration', False)}\n"
                f"- severity={slots.get('has_severity', False)}\n"
                f"- red_flag_check={slots.get('has_red_flags', False)}\n"
                f"缺失欄位：{missing}\n\n"
                "輸出限制：只輸出這三行，不要加編號、不要尖括號、不要多餘解釋：\n"
                "我先確認一件事：<請改成你實際要問的問題，不要保留這段說明文字>\n"
                "若出現麻木無力、胸痛、暈眩、發燒或無法行走，請立即停止並就醫。\n\n"
                f"使用者原句：{query}\n"
                f"歷史上下文：{history}"
            )
        return (
            "You are in clarification-only mode. Do NOT provide a rehab plan yet.\n"
            "Task: ask exactly one high-value follow-up question and do not re-ask known slots.\n"
            "First audit history context and Known slots. Do not ask already-provided fields.\n"
            "If the body area appears to have changed, briefly confirm topic switch and ask whether the previous issue has resolved.\n"
            "Known slots (true means provided):\n"
            f"- body_location={slots.get('has_body', False)}\n"
            f"- trigger_situation={slots.get('has_trigger', False)}\n"
            f"- duration={slots.get('has_duration', False)}\n"
            f"- severity={slots.get('has_severity', False)}\n"
            f"- red_flag_check={slots.get('has_red_flags', False)}\n"
            f"Missing slots: {missing}\n\n"
            "Output restriction: only these 3 lines, no numbering, no angle brackets, no extra explanation:\n"
            "One quick check before I give a precise plan: <replace with your actual question; do not keep this placeholder text>\n"
            "Please answer in one sentence (no A/B format needed).\n"
            "If numbness/weakness, chest pain, dizziness, fever, or inability to walk appears, stop and seek urgent care.\n\n"
            f"User query: {query}\n"
            f"History context: {history}"
        )

    @staticmethod
    def _looks_like_clarify_response(text: str, language: str) -> bool:
        t = text.strip()
        if not t:
            return False
        if len(t) > 1200:
            return False
        if language == "zh":
            if "確認" not in t and "問題" not in t:
                return False
            if "居家計畫" in t or "步驟 1" in t or "問題判讀" in t:
                return False
        else:
            lowered = t.lower()
            if "one quick check" not in lowered and "before i give a precise plan" not in lowered:
                return False
            if "home plan" in lowered or "step 1" in lowered or "problem interpretation" in lowered:
                return False
        return "?" in t or "？" in t

    @staticmethod
    def _is_repeated_clarify_question(text: str, conversation_context: Optional[str]) -> bool:
        if not conversation_context:
            return False
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if not first_line:
            return False
        normalized = re.sub(r"\s+", " ", first_line).lower()
        if len(normalized) < 8:
            return False
        context_norm = re.sub(r"\s+", " ", conversation_context).lower()
        return normalized in context_norm

    @staticmethod
    def _is_redundant_clarify_question(
        text: str,
        *,
        known_slots: Optional[Dict[str, str]] = None,
    ) -> bool:
        if not known_slots:
            return False
        lowered = text.lower()

        checks = [
            ("body_bucket", ["位置", "哪裡", "哪個部位", "where is", "location"]),
            ("trigger", ["什麼情境", "什麼動作", "誘發", "what triggers", "trigger"]),
            ("duration", ["多久", "持續", "how long", "duration"]),
            ("pain_type", ["哪種疼痛", "什麼疼痛", "what type of pain", "pain type"]),
            ("severity", ["幾分", "pain level", "0-10"]),
            ("red_flags", ["麻木無力", "紅旗", "numbness", "red-flag", "red flag"]),
        ]

        for slot_name, patterns in checks:
            if known_slots.get(slot_name) and any(p in text or p in lowered for p in patterns):
                return True
        return False

    def _build_hybrid_clarify_response(
        self,
        *,
        query: str,
        language: str,
        slots: Dict[str, bool],
        conversation_context: Optional[str],
        known_slots: Optional[Dict[str, str]] = None,
    ) -> str:
        if not self.llm_api:
            return self._build_clarify_first_response(
                query=query,
                language=language,
                known_slots=known_slots,
                conversation_context=conversation_context,
            )
        prompt = self._build_dynamic_clarify_prompt(
            query=query,
            language=language,
            slots=slots,
            conversation_context=conversation_context,
            known_slots=known_slots,
        )
        try:
            response = self.llm_api.generate(
                prompt=prompt,
                system=self.system_prompt,
                temperature=0.0,
                max_tokens=260,
                extra=self._gen_extra(),
            )
            text = response.text.strip()
            if self._looks_like_clarify_response(text, language=language):
                if self._is_repeated_clarify_question(text, conversation_context):
                    return self._build_clarify_first_response(
                        query=query,
                        language=language,
                        known_slots=known_slots,
                        conversation_context=conversation_context,
                    )
                if self._is_redundant_clarify_question(text, known_slots=known_slots):
                    return self._build_clarify_first_response(
                        query=query,
                        language=language,
                        known_slots=known_slots,
                        conversation_context=conversation_context,
                    )
                return text
        except Exception:
            pass
        return self._build_clarify_first_response(
            query=query,
            language=language,
            known_slots=known_slots,
            conversation_context=conversation_context,
        )

    @staticmethod
    def _strip_unneeded_clarification_block(answer: str) -> str:
        stripped = answer
        # Remove common leading clarification sections when the query is already specific.
        stripped = re.sub(
            r"先確認.*?(?=(?:\n\s*1\)\s*問題判讀|\n\s*問題判讀|$))",
            "",
            stripped,
            flags=re.DOTALL,
        )
        stripped = re.sub(
            r"Please confirm.*?(?=(?:\n\s*1\)\s*Problem interpretation|\n\s*Problem interpretation|$))",
            "",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return stripped.strip()

    @staticmethod
    def _looks_like_truncated_answer(text: str) -> bool:
        compact = " ".join(text.strip().split())
        if len(compact) < 12:
            return True
        if compact.endswith(("我", "I", "I am", "好的，我", "Okay, I")):
            return True
        return False

    @staticmethod
    def _looks_like_clarification_content(text: str) -> bool:
        lowered = text.lower()
        return (
            "我先確認一件事" in text
            or "先確認" in text
            or "請問" in text
            or "one quick check" in lowered
            or "please confirm" in lowered
        )

    @staticmethod
    def _strip_external_references(text: str) -> str:
        cleaned = MD_LINK_PATTERN.sub(r"\1", text)
        cleaned = URL_PATTERN.sub("", cleaned)
        return cleaned

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        compact = re.sub(r"\s+", " ", text.strip().lower())
        compact = re.sub(r"\[[^\]]+\]", "", compact)
        compact = re.sub(r"references?:.*$", "", compact, flags=re.IGNORECASE | re.DOTALL)
        return compact

    @classmethod
    def _is_repetitive_with_last_answer(cls, answer: str, last_answer: Optional[str]) -> bool:
        if not last_answer:
            return False
        a = cls._normalize_for_similarity(answer)
        b = cls._normalize_for_similarity(last_answer)
        if not a or not b:
            return False
        sa = set(BM25Retriever.tokenize(a))
        sb = set(BM25Retriever.tokenize(b))
        if not sa or not sb:
            return False
        overlap = len(sa & sb) / float(max(1, min(len(sa), len(sb))))
        return overlap >= 0.80

    @staticmethod
    def _find_ungrounded_modalities(answer: str, context_text: str) -> list[str]:
        pairs = [
            ("冰敷", ["冰敷", "ice"]),
            ("熱敷", ["熱敷", "heat"]),
            ("藥物", ["藥物", "藥", "medication", "nsaid"]),
        ]
        answer_l = answer.lower()
        context_l = context_text.lower()
        bad: list[str] = []
        for label, keys in pairs:
            has_in_answer = any(k.lower() in answer_l for k in keys)
            has_in_context = any(k.lower() in context_l for k in keys)
            if has_in_answer and not has_in_context:
                bad.append(label)
        return bad

    def _build_safety_fallback(self, language: str) -> str:
        if language == "zh":
            return (
                "我目前檢索到的資料不足以直接給出精準建議。\n"
                "先給你安全原則：若疼痛持續惡化、出現麻木無力、胸痛或暈眩，請立即停止訓練並就醫。\n"
                "先確認（請回覆最接近的選項）：\n"
                "1) 你是不是在特定動作才會痛？（A推舉/抬手 B轉頭 C彎腰 D走路/跑步）\n"
                "2) 你是不是有神經症狀？（A沒有 B偶爾麻 C持續麻或無力）\n"
                "3) 你是不是已經持續超過72小時或反覆超過2週？（A否 B是）\n"
                "我再根據資訊提供更精準的居家復健流程。"
            )
        return (
            "I don't have enough retrieved evidence to provide a precise plan yet.\n"
            "Safety first: if pain is worsening, or you have numbness/weakness, chest pain, or dizziness, stop training and seek medical care now.\n"
            "Please confirm (choose the closest option):\n"
            "1) Is pain triggered by a specific movement? (A press/lift B turn head C bend D walk/run)\n"
            "2) Any neuro symptoms? (A none B occasional numbness C persistent numbness/weakness)\n"
            "3) Has it lasted >72 hours or recurred for >2 weeks? (A no B yes)\n"
            "Then I will provide a more precise home-rehab flow."
        )

    def _build_user_prompt(
        self,
        query: str,
        context_text: str,
        is_vague_query: bool,
        language: str,
        conversation_context: Optional[str] = None,
        known_slots: Optional[Dict[str, str]] = None,
    ) -> str:
        alt_request = self._is_alternative_request(query)
        if self.response_style == "natural":
            slot_lines = ""
            if known_slots:
                joined = "; ".join([f"{k}={v}" for k, v in sorted(known_slots.items()) if str(v).strip()])
                if joined:
                    slot_lines = f"\nKnown slots (authoritative): {joined}\n"
            alt_rule_zh = ""
            alt_rule_en = ""
            if alt_request:
                alt_rule_zh = (
                    "10) 使用者在問「其他做法/替代方案」：請提供至少 3 個不同機制的替代建議，"
                    "每個都要含適用情境與停止條件，且不得重複同一建議。\n"
                )
                alt_rule_en = (
                    "10) User asked for alternatives: provide at least 3 distinct options with when-to-use and stop conditions; avoid repeating the same advice.\n"
                )
            if language == "zh":
                return (
                    "你是安全優先的居家復健助理。請用自然對話語氣，不要僵硬模板。\n"
                    "規則：\n"
                    "1) 只使用檢索證據，不可捏造。\n"
                    "2) 先簡短回應使用者情境，再給可執行建議。\n"
                    "3) 若資訊不足，最多補問 1 個最關鍵問題，不要一次連問多題。\n"
                    "3.1) 先讀取歷史上下文與 Known slots，禁止重問已提供資訊。\n"
                    "3.2) 若看起來換成新部位，先一句確認是否切換主問題，並問先前問題是否已緩解。\n"
                    "4) 有紅旗症狀時才明確就醫，不要每次都貼同一段固定警語。\n"
                    "5) 重要建議後加上 chunk_id 引用（`[chunk_id]`），最後列 References。\n"
                    "6) 不要輸出固定章節標題（例如 1)問題判讀 2)安全提醒）。\n"
                    "7) 不要使用外部網址作為引用。\n"
                    "8) 不要給萬用模板建議，需依部位與誘發情境給對應建議。\n"
                    "9) 若證據未提及冰敷/熱敷/藥物，不要主動建議。\n\n"
                    "10) 避免與上一輪回答重複措辭；若使用者追問『還有嗎』，必須給不同於上一輪的新做法。\n\n"
                    f"{alt_rule_zh}"
                    f"查詢狀態：vague_query={'yes' if is_vague_query else 'no'}\n"
                    f"{slot_lines}"
                    f"歷史上下文：\n{conversation_context if conversation_context else '（無）'}\n\n"
                    f"使用者問題：\n{query}\n\n"
                    f"檢索內容：\n{context_text if context_text else '（無檢索結果）'}"
                )
            return (
                "You are a safety-first home rehab assistant. Use a natural conversational style.\n"
                "Rules:\n"
                "1) Use retrieved evidence only; do not invent details.\n"
                "2) Briefly reflect the user's situation, then give actionable advice.\n"
                "3) If information is missing, ask at most one key follow-up question.\n"
                "3.1) Read history context and Known slots first; never re-ask already provided details.\n"
                "3.2) If body area appears switched, briefly confirm topic switch and ask whether previous issue has resolved.\n"
                "4) Advise urgent care only when red flags are present; avoid repeating a fixed warning block every turn.\n"
                "5) Add chunk citations (`[chunk_id]`) near key recommendations and end with References.\n"
                "6) Do not use rigid section templates like '1) Problem interpretation'.\n"
                "7) Do not cite external URLs.\n"
                "8) Avoid one-size-fits-all advice; tailor to body area and trigger.\n"
                "9) If evidence does not mention ice/heat/medications, do not suggest them.\n\n"
                "10) Avoid repeating prior answer phrasing; if user asks for alternatives, provide genuinely different options.\n\n"
                f"{alt_rule_en}"
                f"Query state: vague_query={'yes' if is_vague_query else 'no'}\n"
                f"{slot_lines}"
                f"History context:\n{conversation_context if conversation_context else '(none)'}\n\n"
                f"User query:\n{query}\n\n"
                f"Retrieved evidence:\n{context_text if context_text else '(no retrieval results)'}"
            )

        if language == "zh":
            clarification_rules = (
                "釐清規則：\n"
                "A) 若問題描述模糊，先輸出 `先確認` 區塊，提出 2-3 個「你是不是...」封閉式問題（A/B/C）。\n"
                "B) 即使先確認，也要給低風險暫行方案（2-3步），直到使用者補充資訊。\n"
                "C) 若資訊足夠，直接給完整計畫，不要再追問。\n"
                "D) 若使用者已提供部位或情境，禁止重複追問同欄位。\n"
                "E) 當 vague_query=no 時，禁止輸出 `先確認`。\n"
            )
            response_template = (
                "回覆格式（依序）：\n"
                "1) 問題判讀（1-2句）\n"
                "2) 安全提醒（紅旗、何時停止、何時就醫）\n"
                "3) 居家計畫（步驟1/2/3...，每步含次數/頻率/停止條件）\n"
                "4) 進階與回退條件（什麼時候可加量、何時降強度）\n"
                "5) References（列出 chunk_id）\n"
            )
            return (
                "請根據以下檢索到的資料回答使用者問題。\n"
                "規則：\n"
                "1) 先給安全提醒（若有紅旗症狀要立即就醫）。\n"
                "2) 給可執行、具體、可追蹤的居家復健步驟。\n"
                "3) 每一步都要提供頻率/次數/停止條件。\n"
                "4) 每一段建議都必須附上 chunk_id 引用（格式：`[chunk_id]`）。\n"
                "5) 若證據不足，請明確寫「證據不足，需補充資訊」，不要自行補醫療細節。\n"
                "6) 最後用 `References` 列出你實際引用的 chunk_id 清單。\n\n"
                f"{clarification_rules}\n"
                f"{response_template}\n"
                f"此次查詢狀態：vague_query={'yes' if is_vague_query else 'no'}。\n"
                f"使用者歷史上下文：\n{conversation_context if conversation_context else '（無）'}\n\n"
                f"使用者問題：\n{query}\n\n"
                f"檢索內容：\n{context_text if context_text else '（無檢索結果）'}"
            )

        clarification_rules = (
            "Clarification rules:\n"
            "A) If the query is vague, output a `Please confirm` block with 2-3 targeted closed questions (A/B/C).\n"
            "B) Even when asking clarification, provide a low-risk temporary 2-3 step plan.\n"
            "C) If information is sufficient, provide a full plan directly.\n"
            "D) Do not re-ask slots already provided by the user.\n"
            "E) When vague_query=no, do not output `Please confirm`.\n"
        )
        response_template = (
            "Response structure (in order):\n"
            "1) Problem interpretation (1-2 lines)\n"
            "2) Safety reminders (red flags, stop conditions, when to seek care)\n"
            "3) Home plan (Step 1/2/3..., each with frequency/reps/stop condition)\n"
            "4) Progression and deload criteria\n"
            "5) References (chunk_id list)\n"
        )
        return (
            "Answer the user's question using only the retrieved evidence below.\n"
            "Rules:\n"
            "1) Start with safety reminders (urgent care for red-flag symptoms).\n"
            "2) Provide concrete, actionable, trackable home-rehab steps.\n"
            "3) Every step must include dosage/frequency and stop conditions.\n"
            "4) Every recommendation paragraph must cite chunk_id (`[chunk_id]`).\n"
            "5) If evidence is insufficient, explicitly say so; do not invent medical details.\n"
            "6) End with a `References` section listing used chunk_id values.\n"
            "7) Reply in the same language as the user's query.\n\n"
            f"{clarification_rules}\n"
            f"{response_template}\n"
            f"Query state: vague_query={'yes' if is_vague_query else 'no'}.\n"
            f"User history context:\n{conversation_context if conversation_context else '(none)'}\n\n"
            f"User query:\n{query}\n\n"
            f"Retrieved evidence:\n{context_text if context_text else '(no retrieval results)'}"
        )

    def answer(
        self,
        query: str,
        top_k: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 512,
        conversation_context: Optional[str] = None,
        preferred_language: Optional[str] = None,
        force_no_clarify: bool = False,
        known_slots: Optional[Dict[str, str]] = None,
        last_assistant_answer: Optional[str] = None,
    ) -> RAGResult:
        retrieval_k = max(top_k, self.candidate_pool)
        retrieval_query = self._augment_query_with_slots(query, known_slots)
        base_hits, rewrite_notes = self._retrieve_candidates(query=retrieval_query, top_k=retrieval_k)
        body_hits, body_notes = self._apply_body_policy(query=query, hits=base_hits, top_k=top_k)
        safety_hits, safety_notes = self._apply_safety_policy(query=query, hits=body_hits, top_k=top_k)
        chunks, diversity_notes = self._apply_diversity(safety_hits, top_k=top_k)
        policy_notes = rewrite_notes + body_notes + safety_notes + diversity_notes
        if retrieval_query != query:
            policy_notes.append("slot_query_expansion")
        context_text = self._format_context(chunks)
        safety_intent = self._has_safety_intent(query)
        tags_all = {tag for chunk in chunks for tag in chunk.tags}
        expected_body_tags = self._infer_expected_body_tags(query)
        has_expected_body = bool(tags_all & expected_body_tags) if expected_body_tags else any(
            tag.startswith("body_") for tag in tags_all
        )
        has_safety = bool(tags_all & SAFETY_TAGS)
        is_vague_query = self._is_query_vague(query)
        if force_no_clarify and is_vague_query:
            is_vague_query = False
            policy_notes.append("forced_no_clarify")
        query_language = preferred_language if preferred_language in {"zh", "en"} else self._detect_query_language(query)
        if is_vague_query:
            policy_notes.append("clarification_mode")

        if not self.llm_api:
            raise RuntimeError("llm_api is not configured for generation.")

        if self.strict_safety_fallback and safety_intent and (not has_expected_body or not has_safety):
            policy_notes.append("insufficient_evidence_gate")
            fallback_answer = self._build_safety_fallback(query_language)
            return RAGResult(answer=fallback_answer, retrieved=chunks, context_text=context_text, policy_notes=policy_notes)

        if is_vague_query and self.clarify_first_only:
            policy_notes.append("clarify_first_only")
            slots = self._slots_with_known(self._extract_query_slots(query), known_slots)
            if self.clarification_mode == "hybrid":
                policy_notes.append("clarification_mode_hybrid")
                clarify_answer = self._build_hybrid_clarify_response(
                    query=query,
                    language=query_language,
                    slots=slots,
                    conversation_context=conversation_context,
                    known_slots=known_slots,
                )
            else:
                policy_notes.append("clarification_mode_template")
                clarify_answer = self._build_clarify_first_response(
                    query=query,
                    language=query_language,
                    known_slots=known_slots,
                    conversation_context=conversation_context,
                )
            return RAGResult(
                answer=clarify_answer,
                retrieved=chunks,
                context_text=context_text,
                policy_notes=policy_notes,
            )

        user_prompt = self._build_user_prompt(
            query=query,
            context_text=context_text,
            is_vague_query=is_vague_query,
            language=query_language,
            conversation_context=conversation_context,
            known_slots=known_slots,
        )

        response = self.llm_api.generate(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=self._gen_extra(),
        )
        final_answer = response.text
        if self.retry_on_short and self._looks_like_truncated_answer(final_answer):
            retry_prompt = (
                f"{user_prompt}\n\n"
                "上一版回覆過短或被截斷。"
                "請完整回覆，不要只輸出開頭句。"
            )
            retry = self.llm_api.generate(
                prompt=retry_prompt,
                system=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=self._gen_extra(),
            )
            if not self._looks_like_truncated_answer(retry.text):
                final_answer = retry.text
                policy_notes.append("retry_short_answer")
        if force_no_clarify and self._looks_like_clarification_content(final_answer):
            hard_prompt = (
                f"{user_prompt}\n\n"
                "重要：使用者已提供足夠資訊。不要再反問。"
                "請直接提供可執行的居家方案、停止條件與就醫條件。"
            )
            retry2 = self.llm_api.generate(
                prompt=hard_prompt,
                system=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=self._gen_extra(),
            )
            if not self._looks_like_clarification_content(retry2.text):
                final_answer = retry2.text
                policy_notes.append("retry_remove_clarification")
        ungrounded = self._find_ungrounded_modalities(final_answer, context_text)
        if ungrounded:
            retry3_prompt = (
                f"{user_prompt}\n\n"
                f"你剛剛提到未被證據支持的建議：{', '.join(ungrounded)}。"
                "請改寫為只使用檢索證據支持的建議，不要使用這些項目。"
            )
            retry3 = self.llm_api.generate(
                prompt=retry3_prompt,
                system=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                extra=self._gen_extra(),
            )
            final_answer = retry3.text
            policy_notes.append("removed_ungrounded_modalities")
        if self._is_repetitive_with_last_answer(final_answer, last_assistant_answer):
            retry4_prompt = (
                f"{user_prompt}\n\n"
                "你上一輪回答和目前草稿過於重複。"
                "請換不同表達與不同建議順序，並至少提供兩個和上一輪不同的可執行建議。"
            )
            retry4 = self.llm_api.generate(
                prompt=retry4_prompt,
                system=self.system_prompt,
                temperature=min(0.6, max(0.2, temperature + 0.15)),
                max_tokens=max_tokens,
                extra=self._gen_extra(),
            )
            final_answer = retry4.text
            policy_notes.append("retry_reduce_repetition")
        if is_vague_query:
            if self.response_style != "natural":
                final_answer = self._ensure_clarification_block(
                    final_answer,
                    query,
                    language=query_language,
                    known_slots=known_slots,
                    conversation_context=conversation_context,
                )
        else:
            cleaned = self._strip_unneeded_clarification_block(final_answer)
            if cleaned != final_answer:
                policy_notes.append("removed_unneeded_clarification")
            final_answer = cleaned
        final_answer = self._strip_external_references(final_answer)
        final_answer = self._ensure_reference_block(final_answer, chunks)

        return RAGResult(
            answer=final_answer,
            retrieved=chunks,
            context_text=context_text,
            policy_notes=policy_notes,
        )
