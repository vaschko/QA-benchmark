"""Builds the appropriate provider purely from the model configuration."""

from __future__ import annotations

from ..config import ModelConfig
from .base import LLMProvider


def build_provider(cfg: ModelConfig) -> LLMProvider:
    if cfg.provider == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(cfg.model, cfg.options)
    if cfg.provider == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(cfg.model, cfg.options, cfg.api_key_env)
    raise ValueError(f"Unknown provider: {cfg.provider!r}")
