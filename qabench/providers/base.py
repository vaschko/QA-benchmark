"""Abstract provider interface for LLM calls.

Every call is *stateless*: no chat history, no context reused between calls.
This is exactly what guarantees that, when answering a question, only the
explicitly passed context is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMProvider(ABC):
    def __init__(self, model: str, options: Optional[dict[str, Any]] = None):
        self.model = model
        self.options = options or {}

    @abstractmethod
    def generate(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: Optional[dict] = None,
        cacheable: Optional[str] = None,
    ) -> str:
        """A single, stateless completion.

        Args:
            system: System instruction.
            prompt: The variable user request.
            json_schema: If set, structured JSON output is forced according to
                this (flat) schema; the return value is a JSON string.
            cacheable: Large, stable prefix (e.g. the document) placed before
                the prompt. With Anthropic it is reused via prompt caching; with
                Ollama it is simply prepended.

        Returns:
            The raw response text (or a JSON string when a schema is set).
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}(model={self.model!r})"
