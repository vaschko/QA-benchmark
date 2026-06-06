"""OpenAI provider (OpenAI API).

Structured outputs are implemented via forced function calling. OpenAI applies
prompt caching automatically to long, repeated prefixes, so the stable prefix
(`cacheable`, e.g. the document) is simply placed first in the user message.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str,
        options: Optional[dict[str, Any]] = None,
        api_key_env: Optional[str] = None,
    ):
        super().__init__(model, options)
        api_key_env = api_key_env or "OPENAI_API_KEY"
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is missing -- please run `pip install -r requirements.txt`."
            ) from exc

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env} is not set (OpenAI API key)."
            )
        # base_url is picked up from OPENAI_BASE_URL automatically if set,
        # which also allows OpenAI-compatible endpoints.
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: Optional[dict] = None,
        cacheable: Optional[str] = None,
    ) -> str:
        opts = dict(self.options)
        temperature = opts.pop("temperature", None)
        max_tokens = opts.pop("max_tokens", None)

        # Put the stable prefix (document) first so automatic prompt caching hits.
        user_content = f"{cacheable}\n\n{prompt}" if cacheable else prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if json_schema is not None:
            # Structured output via forced function calling.
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "answer",
                        "description": "Return a structured answer according to the schema.",
                        "parameters": json_schema,
                    },
                }
            ]
            kwargs["tool_choice"] = {"type": "function", "function": {"name": "answer"}}
            resp = self._create(kwargs)
            tool_calls = resp.choices[0].message.tool_calls
            if not tool_calls:
                raise RuntimeError("OpenAI returned no tool call.")
            return tool_calls[0].function.arguments

        resp = self._create(kwargs)
        return resp.choices[0].message.content or ""

    def _create(self, kwargs: dict[str, Any]):
        """Call the API; if the model rejects a custom temperature (some GPT-5
        reasoning models only allow the default), retry once without it."""
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                kwargs = {k: v for k, v in kwargs.items() if k != "temperature"}
                return self.client.chat.completions.create(**kwargs)
            raise
