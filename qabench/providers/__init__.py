"""Interchangeable LLM providers (local via Ollama or the Claude API)."""

from .base import LLMProvider
from .factory import build_provider

__all__ = ["LLMProvider", "build_provider"]
