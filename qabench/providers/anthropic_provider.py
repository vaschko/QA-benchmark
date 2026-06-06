"""Anthropic provider (Claude API).

Structured outputs are implemented via forced tool-use. The stable prefix
(`cacheable`, e.g. the document) is marked as its own content block with
cache_control -- with many questions against the same document this saves
noticeable tokens and time (prompt caching).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .base import LLMProvider

DEFAULT_MAX_TOKENS = 2000


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        options: Optional[dict[str, Any]] = None,
        api_key_env: Optional[str] = None,
    ):
        super().__init__(model, options)
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic is missing -- please run `pip install -r requirements.txt`."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set (Anthropic API key)."
            )
        self.client = Anthropic(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: Optional[dict] = None,
        cacheable: Optional[str] = None,
    ) -> str:
        opts = dict(self.options)
        max_tokens = opts.pop("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = opts.pop("temperature", 0.0)

        content: list[dict[str, Any]] = []
        if cacheable:
            content.append(
                {
                    "type": "text",
                    "text": cacheable,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        content.append({"type": "text", "text": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }

        if json_schema is not None:
            # Structured output via forced tool-use.
            kwargs["tools"] = [
                {
                    "name": "answer",
                    "description": "Return a structured answer according to the schema.",
                    "input_schema": json_schema,
                }
            ]
            kwargs["tool_choice"] = {"type": "tool", "name": "answer"}
            resp = self.client.messages.create(**kwargs)
            for block in resp.content:
                if block.type == "tool_use":
                    return json.dumps(block.input)
            raise RuntimeError("Anthropic returned no tool_use block.")

        resp = self.client.messages.create(**kwargs)
        return "".join(b.text for b in resp.content if b.type == "text")
