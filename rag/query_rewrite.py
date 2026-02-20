import re
from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass(frozen=True)
class QueryVariant:
    text: str
    source: str
    weight: float = 1.0


class QueryRewriteProvider(Protocol):
    name: str

    def rewrite(self, query: str) -> List[QueryVariant]:
        ...


class LexiconRewriteProvider:
    """
    Deterministic query expansion layer for mixed zh/en rehab search.
    This is intentionally simple and stable so future providers (LLM translation,
    external terminology service, etc.) can be added without changing pipeline code.
    """

    name = "lexicon"

    def __init__(self) -> None:
        self._rules: List[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"(肩頸|頸肩|neck\s*and\s*shoulder|neck\s+shoulder)", re.IGNORECASE),
                "neck shoulder trapezius scapula cervical posture office syndrome",
            ),
            (
                re.compile(r"(落枕|stiff\s*neck|torticollis)", re.IGNORECASE),
                "acute stiff neck torticollis neck rotation pain stop condition",
            ),
            (
                re.compile(r"(久坐|久站|office|desk|long\s*sitting|sedentary)", re.IGNORECASE),
                "prolonged sitting desk posture ergonomics micro break home exercise",
            ),
            (
                re.compile(r"(手麻|手指麻|麻木|numbness|tingling)", re.IGNORECASE),
                "numbness tingling weakness red flag seek medical care",
            ),
            (
                re.compile(r"(下背|腰痠|腰痛|low\s*back|lumbar)", re.IGNORECASE),
                "low back lumbar pain stiffness extension flexion progression",
            ),
            (
                re.compile(r"(腳踝|ankle|achilles|跟腱|阿基里斯腱)", re.IGNORECASE),
                "ankle achilles load modification return criteria stop condition",
            ),
            (
                re.compile(r"(停止|要不要停|繼續嗎|stop|should i continue|continue exercising)", re.IGNORECASE),
                "stop exercise continue reduce load seek care red flag",
            ),
            (
                re.compile(r"(鍵盤|滑鼠|keyboard|mouse|typing|office\s*work)", re.IGNORECASE),
                "wrist forearm elbow repetitive strain ergonomics micro break tendon nerve glide",
            ),
            (
                re.compile(r"(手腕|前臂|腕|wrist|forearm|carpal|tennis elbow)", re.IGNORECASE),
                "wrist forearm pain numbness tingling load modification office exercise",
            ),
            (
                re.compile(r"(枕頭|睡姿|起床|睡前|morning|bedtime|pillow|sleep posture)", re.IGNORECASE),
                "sleep posture neck shoulder morning routine bedtime routine cervical support stop condition",
            ),
        ]

    def rewrite(self, query: str) -> List[QueryVariant]:
        variants: List[QueryVariant] = []
        for pattern, expansion in self._rules:
            if not pattern.search(query):
                continue
            text = f"{query} {expansion}".strip()
            variants.append(QueryVariant(text=text, source=self.name, weight=0.9))
        return variants


class QueryRewriter:
    def __init__(
        self,
        providers: Optional[List[QueryRewriteProvider]] = None,
        include_original: bool = True,
        max_variants: int = 4,
    ) -> None:
        self.providers = providers or []
        self.include_original = include_original
        self.max_variants = max(1, max_variants)

    def rewrite(self, query: str) -> List[QueryVariant]:
        variants: List[QueryVariant] = []
        seen: set[str] = set()

        def _add(item: QueryVariant) -> None:
            key = " ".join(item.text.lower().split())
            if not key or key in seen:
                return
            seen.add(key)
            variants.append(item)

        if self.include_original:
            _add(QueryVariant(text=query, source="original", weight=1.0))

        for provider in self.providers:
            try:
                for item in provider.rewrite(query):
                    _add(item)
                    if len(variants) >= self.max_variants:
                        return variants
            except Exception:
                # Keep retrieval available even if one provider fails.
                continue

        return variants


def default_query_rewriter() -> QueryRewriter:
    return QueryRewriter(
        providers=[LexiconRewriteProvider()],
        include_original=True,
        max_variants=4,
    )
