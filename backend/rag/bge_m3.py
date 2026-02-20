import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .bm25 import RetrievedChunk


def _fingerprint_docs(docs: List[Dict[str, Any]]) -> str:
    hasher = hashlib.sha1()
    for d in docs:
        hasher.update(str(d.get("chunk_id", "")).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _normalize_scores(scores: Any) -> Any:
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return arr
    max_v = float(arr.max())
    min_v = float(arr.min())
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def _top_k_indices(scores: Any, top_k: int) -> Any:
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=int)
    top_k = min(top_k, arr.size)
    idx = np.argpartition(arr, -top_k)[-top_k:]
    return idx[np.argsort(arr[idx])[::-1]]


class BGEM3DenseRetriever:
    """bge-m3 dense-only retriever via sentence-transformers interface."""

    def __init__(
        self,
        docs: List[Dict[str, Any]],
        embeddings: Any,
        model: Any,
        model_name: str,
    ) -> None:
        self.docs = docs
        self.embeddings = embeddings
        self.model = model
        self.model_name = model_name

    @classmethod
    def from_docs(
        cls,
        docs: List[Dict[str, Any]],
        model_name: str,
        embeddings_path: str,
        device: Optional[str] = None,
    ) -> "BGEM3DenseRetriever":
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(
                "bge-m3 dense retrieval needs `sentence-transformers` and `numpy`. "
                "Install: uv pip install sentence-transformers numpy"
            ) from e

        emb_path = Path(embeddings_path)
        meta_path = emb_path.with_suffix(".meta.json")
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        fp = _fingerprint_docs(docs)
        model = SentenceTransformer(model_name, device=device)

        use_cache = False
        if emb_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("retriever_type") == "bge_m3_dense"
                    and meta.get("model_name") == model_name
                    and meta.get("fingerprint") == fp
                    and meta.get("num_docs") == len(docs)
                ):
                    use_cache = True
            except Exception:
                use_cache = False

        if use_cache:
            embeddings = np.load(str(emb_path))
        else:
            texts = [str(d.get("text", "")) for d in docs]
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
            )
            np.save(str(emb_path), embeddings)
            meta_path.write_text(
                json.dumps(
                    {
                        "retriever_type": "bge_m3_dense",
                        "model_name": model_name,
                        "fingerprint": fp,
                        "num_docs": len(docs),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return cls(docs=docs, embeddings=embeddings, model=model, model_name=model_name)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        scores = self.embeddings @ q_emb
        idx = _top_k_indices(scores, top_k)

        out: List[RetrievedChunk] = []
        for i in idx:
            d = self.docs[int(i)]
            out.append(
                RetrievedChunk(
                    score=float(scores[int(i)]),
                    chunk_id=str(d.get("chunk_id", "")),
                    text=str(d.get("text", "")),
                    source_name=str(d.get("source_name", "")),
                    page=int(d.get("page", 0) or 0),
                    title=str(d.get("title", "")),
                    tags=list(d.get("tags", []) or []),
                )
            )
        return out


class BGEM3DenseSparseRetriever:
    """bge-m3 retriever using dense + sparse (lexical) score fusion."""

    def __init__(
        self,
        docs: List[Dict[str, Any]],
        dense_embeddings: Any,
        sparse_embeddings: List[Dict[str, float]],
        model: Any,
        model_name: str,
        dense_weight: float = 0.5,
    ) -> None:
        self.docs = docs
        self.dense_embeddings = dense_embeddings
        self.sparse_embeddings = sparse_embeddings
        self.model = model
        self.model_name = model_name
        self.dense_weight = dense_weight

    @classmethod
    def from_docs(
        cls,
        docs: List[Dict[str, Any]],
        model_name: str,
        embeddings_path: str,
        dense_weight: float = 0.5,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 2048,
    ) -> "BGEM3DenseSparseRetriever":
        try:
            import numpy as np
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:
            raise RuntimeError(
                "bge-m3 dense+sparse needs `FlagEmbedding` and `numpy`. "
                "Install: uv pip install FlagEmbedding numpy"
            ) from e

        emb_path = Path(embeddings_path)
        sparse_path = emb_path.with_suffix(".sparse.pkl")
        meta_path = emb_path.with_suffix(".meta.json")
        emb_path.parent.mkdir(parents=True, exist_ok=True)

        fp = _fingerprint_docs(docs)
        use_fp16 = device not in {"cpu", "mps"}
        model = BGEM3FlagModel(model_name, use_fp16=use_fp16, device=device)

        use_cache = False
        if emb_path.exists() and sparse_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    meta.get("retriever_type") == "bge_m3_dense_sparse"
                    and meta.get("model_name") == model_name
                    and meta.get("fingerprint") == fp
                    and meta.get("num_docs") == len(docs)
                ):
                    use_cache = True
            except Exception:
                use_cache = False

        if use_cache:
            dense_embeddings = np.load(str(emb_path))
            with sparse_path.open("rb") as f:
                sparse_embeddings = pickle.load(f)
        else:
            texts = [str(d.get("text", "")) for d in docs]
            encoded = model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense_embeddings = np.asarray(encoded["dense_vecs"], dtype=float)
            sparse_embeddings = encoded["lexical_weights"]

            np.save(str(emb_path), dense_embeddings)
            with sparse_path.open("wb") as f:
                pickle.dump(sparse_embeddings, f)

            meta_path.write_text(
                json.dumps(
                    {
                        "retriever_type": "bge_m3_dense_sparse",
                        "model_name": model_name,
                        "fingerprint": fp,
                        "num_docs": len(docs),
                        "dense_weight": dense_weight,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        return cls(
            docs=docs,
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            model=model,
            model_name=model_name,
            dense_weight=dense_weight,
        )

    def _score_components(self, query: str) -> Any:
        import numpy as np

        encoded = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        q_dense = np.asarray(encoded["dense_vecs"], dtype=float)[0]
        q_sparse = encoded["lexical_weights"][0]

        dense_scores = self.dense_embeddings @ q_dense
        sparse_scores = np.asarray(
            [
                float(self.model.compute_lexical_matching_score(q_sparse, doc_sparse))
                for doc_sparse in self.sparse_embeddings
            ],
            dtype=float,
        )
        return dense_scores, sparse_scores

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        dense_scores, sparse_scores = self._score_components(query)
        dense_norm = _normalize_scores(dense_scores)
        sparse_norm = _normalize_scores(sparse_scores)

        scores = self.dense_weight * dense_norm + (1.0 - self.dense_weight) * sparse_norm
        idx = _top_k_indices(scores, top_k)

        out: List[RetrievedChunk] = []
        for i in idx:
            d = self.docs[int(i)]
            out.append(
                RetrievedChunk(
                    score=float(scores[int(i)]),
                    chunk_id=str(d.get("chunk_id", "")),
                    text=str(d.get("text", "")),
                    source_name=str(d.get("source_name", "")),
                    page=int(d.get("page", 0) or 0),
                    title=str(d.get("title", "")),
                    tags=list(d.get("tags", []) or []),
                )
            )
        return out
