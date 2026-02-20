import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Set
from urllib import error, request
from urllib.parse import quote_plus, urlencode

from backend.rag.pipeline import BODY_HINT_PATTERNS, SAFETY_QUERY_PATTERNS


CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def detect_query_language(query: str) -> str:
    return "zh" if CJK_PATTERN.search(query) else "en"


def infer_body_tags(query: str) -> Set[str]:
    tags: Set[str] = set()
    for tag, pattern in BODY_HINT_PATTERNS.items():
        if pattern.search(query):
            tags.add(tag)
    return tags


def infer_intent_tags(query: str) -> Set[str]:
    tags: Set[str] = set()
    if any(pattern.search(query) for pattern in SAFETY_QUERY_PATTERNS):
        tags.add("safety_intent")
    return tags


ZH_BODY_LABELS: Dict[str, str] = {
    "body_neck_trap": "肩頸/斜方肌",
    "body_shoulder": "肩關節/旋轉肌袖",
    "body_back_spine": "下背/脊椎",
    "body_elbow_wrist_hand": "手肘/手腕/前臂",
    "body_knee": "膝關節",
    "body_ankle_foot": "踝足",
    "body_hip_glute": "髖臀",
}

EN_BODY_LABELS: Dict[str, str] = {
    "body_neck_trap": "neck/trapezius",
    "body_shoulder": "shoulder/rotator cuff",
    "body_back_spine": "low back/spine",
    "body_elbow_wrist_hand": "elbow/wrist/forearm",
    "body_knee": "knee",
    "body_ankle_foot": "ankle/foot",
    "body_hip_glute": "hip/glute",
}


def _tokenize(text: str) -> Set[str]:
    return set(TOKEN_PATTERN.findall(text.lower()))


@dataclass
class VideoCandidate:
    video_id: str
    title: str
    url: str
    provider: str
    language: str = "mixed"
    tags: List[str] = field(default_factory=list)
    intent_tags: List[str] = field(default_factory=list)
    difficulty: str = "general"
    notes: str = ""
    why: List[str] = field(default_factory=list)
    score: float = 0.0


def summarize_video(candidate: VideoCandidate, language: str = "zh") -> str:
    if language == "zh":
        body_labels = [ZH_BODY_LABELS.get(tag, tag) for tag in candidate.tags if tag.startswith("body_")]
        body_text = "、".join(body_labels) if body_labels else "一般活動與恢復"
        safety = "。含安全提醒，若症狀惡化請停止並就醫" if "safety_intent" in candidate.intent_tags else ""
        base = f"重點針對 {body_text}，難度 {candidate.difficulty}，適合作為居家跟練參考"
        if candidate.notes:
            return f"{base}。{candidate.notes}{safety}"
        return f"{base}{safety}"

    body_labels = [EN_BODY_LABELS.get(tag, tag) for tag in candidate.tags if tag.startswith("body_")]
    body_text = ", ".join(body_labels) if body_labels else "general mobility and recovery"
    safety = ". Includes safety reminders; stop and seek care if symptoms worsen" if "safety_intent" in candidate.intent_tags else ""
    base = f"Focuses on {body_text}, difficulty={candidate.difficulty}, suitable for guided home practice"
    if candidate.notes:
        return f"{base}. {candidate.notes}{safety}"
    return f"{base}{safety}"


class VideoProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        body_tags: Set[str],
        intent_tags: Set[str],
        language: str,
        limit: int,
    ) -> List[VideoCandidate]:
        ...


class CatalogVideoProvider:
    name = "catalog"

    def __init__(self, catalog_path: str) -> None:
        self.catalog_path = Path(catalog_path)
        self.items = self._load_catalog(self.catalog_path)

    @staticmethod
    def _load_catalog(path: Path) -> List[Dict[str, object]]:
        if not path.exists():
            return []
        items: List[Dict[str, object]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("title") or not row.get("url"):
                    continue
                items.append(row)
        return items

    def search(
        self,
        query: str,
        body_tags: Set[str],
        intent_tags: Set[str],
        language: str,
        limit: int,
    ) -> List[VideoCandidate]:
        if not self.items:
            return []

        query_tokens = _tokenize(query)
        ranked: List[VideoCandidate] = []

        for row in self.items:
            title = str(row.get("title", ""))
            tags = list(row.get("tags", []) or [])
            item_intents = list(row.get("intent_tags", []) or [])
            aliases = list(row.get("aliases", []) or [])
            item_lang = str(row.get("language", "mixed"))

            haystack = " ".join([title] + aliases)
            overlap = len(query_tokens & _tokenize(haystack))
            body_overlap = len(body_tags & set(tags))
            intent_overlap = len(intent_tags & set(item_intents))

            score = 0.0
            score += overlap * 0.35
            score += body_overlap * 1.6
            score += intent_overlap * 1.2
            if item_lang in {"mixed", language}:
                score += 0.6

            # If query already has clear body tags, avoid cross-body noise.
            if body_tags and body_overlap == 0 and overlap == 0 and intent_overlap == 0:
                continue

            if score <= 0.2:
                continue

            why: List[str] = []
            if body_overlap > 0:
                why.append(f"body_match={body_overlap}")
            if intent_overlap > 0:
                why.append(f"intent_match={intent_overlap}")
            if overlap > 0:
                why.append(f"keyword_overlap={overlap}")
            if item_lang == language:
                why.append("language_match")

            ranked.append(
                VideoCandidate(
                    video_id=str(row.get("video_id", row.get("id", title))),
                    title=title,
                    url=str(row.get("url")),
                    provider=self.name,
                    language=item_lang,
                    tags=tags,
                    intent_tags=item_intents,
                    difficulty=str(row.get("difficulty", "general")),
                    notes=str(row.get("notes", "")),
                    why=why,
                    score=score,
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:limit]


class YouTubeSearchProvider:
    """
    Fallback provider that returns stable YouTube search links.
    This is crawler/API-friendly: replace query templates with API calls later.
    """

    name = "youtube_search"

    def __init__(self) -> None:
        self.zh_templates: Dict[str, str] = {
            "body_neck_trap": "頸肩 放鬆 伸展 物理治療 居家",
            "body_shoulder": "肩膀 旋轉肌袖 居家復健 教學",
            "body_back_spine": "下背痛 核心穩定 居家運動",
            "body_elbow_wrist_hand": "手腕 前臂 痠麻 神經滑動 居家",
            "body_knee": "膝蓋痛 居家復健 運動",
            "body_ankle_foot": "腳踝 扭傷 居家復健 教學",
            "body_hip_glute": "髖關節 臀肌 居家復健",
        }
        self.en_templates: Dict[str, str] = {
            "body_neck_trap": "neck shoulder mobility physical therapy home exercises",
            "body_shoulder": "rotator cuff shoulder rehab home exercise tutorial",
            "body_back_spine": "low back pain core stability home rehab",
            "body_elbow_wrist_hand": "wrist forearm nerve glide home exercise",
            "body_knee": "knee pain rehabilitation home exercise",
            "body_ankle_foot": "ankle sprain rehab home exercise",
            "body_hip_glute": "hip glute rehabilitation home exercise",
        }

    def _build_queries(self, query: str, body_tags: Set[str], language: str) -> List[str]:
        templates = self.zh_templates if language == "zh" else self.en_templates
        results: List[str] = []
        for tag in body_tags:
            q = templates.get(tag)
            if q:
                results.append(q)
        if not results:
            if language == "zh":
                results.append(f"{query} 居家復健 教學")
            else:
                results.append(f"{query} home rehabilitation exercise tutorial")
        return results

    def search(
        self,
        query: str,
        body_tags: Set[str],
        intent_tags: Set[str],
        language: str,
        limit: int,
    ) -> List[VideoCandidate]:
        queries = self._build_queries(query, body_tags, language)
        candidates: List[VideoCandidate] = []
        for i, q in enumerate(queries[:limit], start=1):
            url = f"https://www.youtube.com/results?search_query={quote_plus(q)}"
            notes = "Use trusted medical/PT channels and verify credentials before following."
            candidates.append(
                VideoCandidate(
                    video_id=f"yt_search_{i}",
                    title=q,
                    url=url,
                    provider=self.name,
                    language=language,
                    tags=sorted(body_tags),
                    intent_tags=sorted(intent_tags),
                    difficulty="general",
                    notes=notes,
                    why=["fallback_search_query"],
                    score=0.2,
                )
            )
        return candidates


def _parse_iso8601_duration(duration: str) -> str:
    # Example: PT7M31S, PT42S, PT1H2M
    m = re.fullmatch(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration or "")
    if not m:
        return "unknown"
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    if h > 0:
        return f"{h:d}:{mi:02d}:{s:02d}"
    return f"{mi:d}:{s:02d}"


class YouTubeDataApiProvider:
    name = "youtube_api"

    def __init__(
        self,
        api_key: Optional[str] = None,
        region_code: str = "TW",
        safe_search: str = "strict",
        channel_whitelist: Optional[List[str]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY", "")
        self.region_code = region_code
        self.safe_search = safe_search
        self.channel_whitelist = set(channel_whitelist or [])

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())

    def _request_json(self, endpoint: str, params: Dict[str, str]) -> Dict[str, object]:
        if not self.enabled:
            raise RuntimeError("YouTube API key is not configured.")
        params = dict(params)
        params["key"] = self.api_key.strip()
        url = f"https://www.googleapis.com/youtube/v3/{endpoint}?{urlencode(params)}"
        req = request.Request(url, method="GET")
        try:
            with request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"YouTube API HTTP {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"YouTube API request failed: {e}") from e

    def _build_search_query(self, query: str, body_tags: Set[str], language: str) -> str:
        if language == "zh":
            mapping = {
                "body_neck_trap": "頸肩 伸展 物理治療 居家",
                "body_shoulder": "肩膀 旋轉肌袖 居家復健",
                "body_back_spine": "下背痛 核心穩定 居家",
                "body_elbow_wrist_hand": "手腕 前臂 神經滑動 居家",
                "body_knee": "膝蓋 居家復健 運動",
                "body_ankle_foot": "腳踝 扭傷 居家復健",
                "body_hip_glute": "髖 臀肌 居家復健",
            }
            for tag in body_tags:
                if tag in mapping:
                    return mapping[tag]
            return f"{query} 居家復健 物理治療"

        mapping = {
            "body_neck_trap": "neck shoulder physical therapy home exercises",
            "body_shoulder": "rotator cuff shoulder rehab home exercises",
            "body_back_spine": "low back pain home rehab core stability",
            "body_elbow_wrist_hand": "wrist forearm nerve glide home exercises",
            "body_knee": "knee rehabilitation home exercises",
            "body_ankle_foot": "ankle sprain home rehab exercises",
            "body_hip_glute": "hip glute home rehabilitation exercises",
        }
        for tag in body_tags:
            if tag in mapping:
                return mapping[tag]
        return f"{query} home rehabilitation exercises"

    def _fetch_video_meta(self, video_ids: List[str]) -> Dict[str, Dict[str, object]]:
        if not video_ids:
            return {}
        payload = self._request_json(
            endpoint="videos",
            params={
                "part": "snippet,contentDetails,statistics",
                "id": ",".join(video_ids),
                "maxResults": str(min(len(video_ids), 50)),
            },
        )
        out: Dict[str, Dict[str, object]] = {}
        for item in payload.get("items", []) or []:
            vid = str(item.get("id", ""))
            if vid:
                out[vid] = item
        return out

    def search(
        self,
        query: str,
        body_tags: Set[str],
        intent_tags: Set[str],
        language: str,
        limit: int,
    ) -> List[VideoCandidate]:
        if not self.enabled:
            return []

        relevance_language = "zh-TW" if language == "zh" else "en"
        search_query = self._build_search_query(query, body_tags, language)
        payload = self._request_json(
            endpoint="search",
            params={
                "part": "snippet",
                "type": "video",
                "q": search_query,
                "maxResults": str(min(max(limit * 3, 8), 25)),
                "safeSearch": self.safe_search,
                "regionCode": self.region_code,
                "relevanceLanguage": relevance_language,
            },
        )

        items = payload.get("items", []) or []
        video_ids: List[str] = []
        snippets_by_id: Dict[str, Dict[str, object]] = {}
        for item in items:
            id_obj = item.get("id", {}) or {}
            video_id = str(id_obj.get("videoId", ""))
            snippet = item.get("snippet", {}) or {}
            if not video_id:
                continue
            channel_id = str(snippet.get("channelId", ""))
            if self.channel_whitelist and channel_id not in self.channel_whitelist:
                continue
            video_ids.append(video_id)
            snippets_by_id[video_id] = snippet

        if not video_ids:
            return []

        meta_by_id = self._fetch_video_meta(video_ids)
        query_tokens = _tokenize(query)
        ranked: List[VideoCandidate] = []
        for idx, video_id in enumerate(video_ids, start=1):
            snippet = snippets_by_id.get(video_id, {})
            title = str(snippet.get("title", ""))
            desc = str(snippet.get("description", ""))
            channel = str(snippet.get("channelTitle", ""))

            meta = meta_by_id.get(video_id, {})
            content_details = meta.get("contentDetails", {}) if isinstance(meta, dict) else {}
            stats = meta.get("statistics", {}) if isinstance(meta, dict) else {}
            duration_raw = str(content_details.get("duration", "")) if isinstance(content_details, dict) else ""
            duration = _parse_iso8601_duration(duration_raw)
            views = str(stats.get("viewCount", "")) if isinstance(stats, dict) else ""

            overlap = len(query_tokens & _tokenize(f"{title} {desc}"))
            score = 0.0
            score += max(0.0, (limit * 3 - idx) * 0.08)
            score += overlap * 0.25
            score += len(body_tags) * 0.3
            if intent_tags:
                score += 0.2

            notes = f"channel={channel}; duration={duration}"
            if views:
                notes += f"; views={views}"

            why = ["youtube_api_search"]
            if overlap > 0:
                why.append(f"keyword_overlap={overlap}")
            if body_tags:
                why.append("body_aligned_query")

            ranked.append(
                VideoCandidate(
                    video_id=video_id,
                    title=title,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                    provider=self.name,
                    language=language,
                    tags=sorted(body_tags),
                    intent_tags=sorted(intent_tags),
                    difficulty="general",
                    notes=notes,
                    why=why,
                    score=score,
                )
            )

        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:limit]


class VideoRecommender:
    def __init__(self, providers: Iterable[VideoProvider]) -> None:
        self.providers = list(providers)

    def recommend(
        self,
        query: str,
        body_tags: Optional[Set[str]] = None,
        intent_tags: Optional[Set[str]] = None,
        language: Optional[str] = None,
        limit: int = 5,
    ) -> List[VideoCandidate]:
        body_tags = body_tags or infer_body_tags(query)
        intent_tags = intent_tags or infer_intent_tags(query)
        language = language or detect_query_language(query)

        results: List[VideoCandidate] = []
        seen_urls: Set[str] = set()
        for provider in self.providers:
            try:
                provider_hits = provider.search(
                    query=query,
                    body_tags=body_tags,
                    intent_tags=intent_tags,
                    language=language,
                    limit=max(limit, 8),
                )
            except Exception:
                # Keep recommendation flow alive even if one provider is down or misconfigured.
                continue
            for hit in provider_hits:
                if hit.url in seen_urls:
                    continue
                seen_urls.add(hit.url)
                results.append(hit)

        # Global rerank after provider merge.
        results.sort(key=lambda x: x.score, reverse=True)
        if len(results) < limit:
            return results
        return results[:limit]
