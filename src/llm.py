"""Thin Ollama HTTP client. Streams tokens via the /api/chat endpoint (NDJSON)."""
from __future__ import annotations

import json
from typing import Iterator

import httpx


class Ollama:
    def __init__(self, host: str, model: str, temperature: float = 0.2, num_ctx: int = 4096):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx

    def health(self) -> tuple[bool, str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=5)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"
            tags = r.json().get("models", [])
            names = [m.get("name", "") for m in tags]
            if not any(n == self.model or n.startswith(f"{self.model}:") for n in names):
                return True, f"connected; model '{self.model}' not pulled yet"
            return True, "ok"
        except Exception as e:
            return False, str(e)

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        with httpx.stream("POST", f"{self.host}/api/chat", json=payload, timeout=None) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = obj.get("message") or {}
                token = msg.get("content")
                if token:
                    yield token
                if obj.get("done"):
                    break
