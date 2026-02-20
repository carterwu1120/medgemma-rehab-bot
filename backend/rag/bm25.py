import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List


EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "as",
    "from",
    "into",
    "about",
    "you",
    "your",
    "i",
    "we",
    "they",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    text: str
    source_name: str
    page: int
    title: str
    tags: List[str]


class BM25Retriever:
    def __init__(
        self,
        docs: List[Dict[str, Any]],
        doc_tfs: List[Counter],
        doc_lengths: List[int],
        df: Dict[str, int],
        avgdl: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.docs = docs
        self.doc_tfs = doc_tfs
        self.doc_lengths = doc_lengths
        self.df = df
        self.avgdl = avgdl
        self.k1 = k1
        self.b = b
        self.N = len(docs)

        self.idf: Dict[str, float] = {}
        for token, freq in self.df.items():
            # BM25 idf with +1 to keep positive values.
            self.idf[token] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def _tokenize_zh_sequence(seq: str) -> List[str]:
        if len(seq) == 1:
            return [seq]
        grams = [seq[i : i + 2] for i in range(len(seq) - 1)]
        # Keep original full sequence as a token for phrase match effect.
        grams.append(seq)
        return grams

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        tokens: List[str] = []
        for raw in TOKEN_PATTERN.findall(text.lower()):
            if not raw:
                continue
            if re.fullmatch(r"[a-z0-9]+", raw):
                if raw in EN_STOPWORDS:
                    continue
                if len(raw) <= 1:
                    continue
                tokens.append(raw)
            else:
                tokens.extend(cls._tokenize_zh_sequence(raw))
        return tokens

    @classmethod
    def from_jsonl(
        cls,
        canonical_path: str,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Retriever":
        path = Path(canonical_path)
        if not path.exists():
            raise FileNotFoundError(f"Canonical docs not found: {path}")

        docs: List[Dict[str, Any]] = []
        doc_tfs: List[Counter] = []
        doc_lengths: List[int] = []
        df_counter: Counter = Counter()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = str(doc.get("text", ""))
                tokens = cls.tokenize(text)
                if not tokens:
                    continue

                tf = Counter(tokens)
                docs.append(doc)
                doc_tfs.append(tf)
                doc_lengths.append(sum(tf.values()))

                for tok in tf.keys():
                    df_counter[tok] += 1

        if not docs:
            raise ValueError("No valid documents found after tokenization.")

        avgdl = sum(doc_lengths) / len(doc_lengths)
        return cls(
            docs=docs,
            doc_tfs=doc_tfs,
            doc_lengths=doc_lengths,
            df=dict(df_counter),
            avgdl=avgdl,
            k1=k1,
            b=b,
        )

    def save(self, out_path: str) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "docs": self.docs,
            "doc_tfs": [dict(tf) for tf in self.doc_tfs],
            "doc_lengths": self.doc_lengths,
            "df": self.df,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, index_path: str) -> "BM25Retriever":
        path = Path(index_path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        with path.open("rb") as f:
            payload = pickle.load(f)

        return cls(
            docs=payload["docs"],
            doc_tfs=[Counter(tf) for tf in payload["doc_tfs"]],
            doc_lengths=payload["doc_lengths"],
            df=payload["df"],
            avgdl=payload["avgdl"],
            k1=payload.get("k1", 1.5),
            b=payload.get("b", 0.75),
        )

    def _score_doc(self, query_terms: Counter, doc_idx: int) -> float:
        tf = self.doc_tfs[doc_idx]
        dl = self.doc_lengths[doc_idx]
        score = 0.0

        for term, qf in query_terms.items():
            if term not in tf:
                continue
            idf = self.idf.get(term, 0.0)
            term_tf = tf[term]
            denom = term_tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += qf * idf * (term_tf * (self.k1 + 1)) / max(denom, 1e-9)

        return score

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> List[RetrievedChunk]:
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        q_tf = Counter(q_tokens)
        scores: List[tuple[int, float]] = []
        for i in range(self.N):
            s = self._score_doc(q_tf, i)
            if s > min_score:
                scores.append((i, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[RetrievedChunk] = []

        for idx, score in scores[:top_k]:
            d = self.docs[idx]
            results.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=str(d.get("chunk_id", "")),
                    text=str(d.get("text", "")),
                    source_name=str(d.get("source_name", "")),
                    page=int(d.get("page", 0) or 0),
                    title=str(d.get("title", "")),
                    tags=list(d.get("tags", []) or []),
                )
            )

        return results

    def stats(self) -> Dict[str, Any]:
        tag_counter: Counter = Counter()
        for doc in self.docs:
            for tag in doc.get("tags", []) or []:
                tag_counter[tag] += 1

        return {
            "num_docs": self.N,
            "vocab_size": len(self.df),
            "avg_doc_len": round(self.avgdl, 2),
            "top_tags": tag_counter.most_common(20),
        }
