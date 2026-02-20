from dataclasses import dataclass
from typing import List, Optional

from llm_api import VLLMApi

from .bm25 import BM25Retriever, RetrievedChunk


SYSTEM_PROMPT = (
    "You are a cautious home rehabilitation assistant. "
    "Always prioritize safety. Do not diagnose. "
    "If red-flag symptoms appear, advise urgent medical care. "
    "Answer in Traditional Chinese unless user asks otherwise."
)


@dataclass
class RAGResult:
    answer: str
    retrieved: List[RetrievedChunk]
    context_text: str


class RehabRAG:
    def __init__(
        self,
        retriever: BM25Retriever,
        llm_api: Optional[VLLMApi] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm_api = llm_api
        self.system_prompt = system_prompt

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        return self.retriever.search(query=query, top_k=top_k)

    @staticmethod
    def _format_context(chunks: List[RetrievedChunk]) -> str:
        lines: List[str] = []
        for i, c in enumerate(chunks, start=1):
            tags = ", ".join(c.tags) if c.tags else "none"
            lines.append(
                f"[{i}] source={c.source_name} page={c.page} chunk_id={c.chunk_id} tags={tags}\n{c.text}"
            )
        return "\n\n".join(lines)

    def answer(self, query: str, top_k: int = 5, temperature: float = 0.2, max_tokens: int = 512) -> RAGResult:
        chunks = self.retrieve(query=query, top_k=top_k)
        context_text = self._format_context(chunks)

        if not self.llm_api:
            raise RuntimeError("llm_api is not configured for generation.")

        user_prompt = (
            "請根據以下檢索到的資料回答使用者問題。\n"
            "規則：\n"
            "1) 先給安全提醒（若有紅旗症狀要立即就醫）。\n"
            "2) 給可執行的居家復健步驟。\n"
            "3) 盡量提供頻率/次數/停止條件。\n"
            "4) 最後用 `References` 列出你實際引用的來源編號（例如 [1], [3]）。\n\n"
            f"使用者問題：\n{query}\n\n"
            f"檢索內容：\n{context_text if context_text else '（無檢索結果）'}"
        )

        response = self.llm_api.generate(
            prompt=user_prompt,
            system=self.system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return RAGResult(answer=response.text, retrieved=chunks, context_text=context_text)
