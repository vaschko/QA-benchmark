"""Stage 2: create the summary (the model under test)."""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..providers import LLMProvider


def make_summary(text: str, cfg: Config, provider: LLMProvider, language: str) -> str:
    system, user = prompts.summarize(cfg.summary.target_words, language)
    raw = provider.generate(system=system, prompt=user, cacheable=text)
    return raw.strip()
