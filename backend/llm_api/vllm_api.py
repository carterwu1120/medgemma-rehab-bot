import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    raw: Dict[str, Any]


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class VLLMApi:
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
        dotenv_path: Optional[str] = None,
    ) -> None:
        if dotenv_path:
            _load_dotenv(Path(dotenv_path))
        else:
            project_root = Path(__file__).resolve().parents[1]
            _load_dotenv(project_root / ".env")

        self.base_url = (base_url or os.getenv("VLLM_BASE_URL") or "http://127.0.0.1:8000").rstrip("/")
        self.model = model or os.getenv("VLLM_MODEL")
        self.timeout = timeout
        self.provider = "vllm"

        if not self.model:
            raise ValueError("VLLM model is not set. Provide `model` or set VLLM_MODEL in .env")

    def _request_json(self, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = {}
        data = None

        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = request.Request(url, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"vLLM HTTP {e.code}: {body}") from e
        except Exception as e:
            raise RuntimeError(f"vLLM request failed at {url}: {e}") from e

    def health_check(self) -> bool:
        try:
            data = self.list_models()
            model_ids = {item.get("id") for item in data.get("data", [])}
            return self.model in model_ids or len(model_ids) > 0
        except Exception:
            return False

    def list_models(self) -> Dict[str, Any]:
        return self._request_json("GET", "/v1/models")

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra:
            payload.update(extra)

        raw = self._request_json("POST", "/v1/chat/completions", payload)
        text = raw["choices"][0]["message"]["content"]
        return LLMResponse(text=text, model=self.model, provider=self.provider, raw=raw)

    def generate(
        self,
        prompt: str,
        system: str = "You are a cautious home rehabilitation assistant. Include safety warnings when needed.",
        temperature: float = 0.2,
        max_tokens: int = 512,
        extra: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )


def create_vllm_api(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
    dotenv_path: Optional[str] = None,
) -> VLLMApi:
    return VLLMApi(model=model, base_url=base_url, timeout=timeout, dotenv_path=dotenv_path)


def generate_text(
    prompt: str,
    system: str = "You are a cautious home rehabilitation assistant. Include safety warnings when needed.",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 120,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    api = create_vllm_api(model=model, base_url=base_url, timeout=timeout)
    return api.generate(
        prompt=prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    ).text
