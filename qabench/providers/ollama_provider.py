"""Ollama provider (local).

Uses the /api/generate endpoint with stream=false and WITHOUT the `context`
field. This makes every call fully stateless -- nothing from previous calls
stays "in memory".
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

from .base import LLMProvider

DEFAULT_TIMEOUT = 600  # local models can be slow


class OllamaProvider(LLMProvider):
    def __init__(self, model: str, options: Optional[dict[str, Any]] = None, host: Optional[str] = None):
        super().__init__(model, options)
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if not host.startswith("http"):
            host = f"http://{host}"
        self.base_url = host.rstrip("/")

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: Optional[dict] = None,
        cacheable: Optional[str] = None,
    ) -> str:
        # Ollama has no separate caching -- simply prepend the stable prefix.
        full_prompt = f"{cacheable}\n\n{prompt}" if cacheable else prompt

        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": full_prompt,
            "stream": False,
            # Disable thinking: otherwise reasoning models put the (structured)
            # output into the 'thinking' field instead of 'response'.
            "think": False,
            "options": dict(self.options),
        }
        if json_schema is not None:
            payload["format"] = json_schema

        resp = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        # Fallback: some models ignore think=False and keep writing to the
        # thinking field -- read from there in that case.
        return data.get("response") or data.get("thinking") or ""
