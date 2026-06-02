"""Load and validate the YAML configuration via pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    provider: Literal["ollama", "anthropic"] = "ollama"
    model: str
    options: dict[str, Any] = Field(default_factory=dict)
    # Only for provider=anthropic: name of the env var holding the API key.
    api_key_env: str = "ANTHROPIC_API_KEY"


class ModelsConfig(BaseModel):
    strong: ModelConfig       # question generation + judge
    summarizer: ModelConfig   # the model under test
    answerer: ModelConfig     # constant across both conditions


class QuestionsConfig(BaseModel):
    count: int = 12
    types: list[str] = Field(default_factory=lambda: ["factual", "numeric", "inferential"])


class SummaryConfig(BaseModel):
    mode: Literal["generate", "file"] = "generate"
    target_words: int = 300


class AnsweringConfig(BaseModel):
    closed_book_baseline: bool = True
    require_evidence: bool = True
    max_context_chars: int = 24000


class JudgeConfig(BaseModel):
    scale: list[str] = Field(default_factory=lambda: ["match", "partial", "mismatch"])


class RunConfig(BaseModel):
    output_dir: str = "runs"


class Config(BaseModel):
    models: ModelsConfig
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    answering: AnsweringConfig = Field(default_factory=AnsweringConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    run: RunConfig = Field(default_factory=RunConfig)


def load_config(path: str | Path) -> Config:
    """Reads the YAML file and returns a validated Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(**data)
