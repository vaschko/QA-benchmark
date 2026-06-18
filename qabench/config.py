"""Load and validate the YAML configuration via pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    provider: Literal["ollama", "anthropic", "openai"] = "ollama"
    model: str
    options: dict[str, Any] = Field(default_factory=dict)
    # Name of the env var holding the API key (for anthropic/openai). If None,
    # each provider uses its own default (ANTHROPIC_API_KEY / OPENAI_API_KEY).
    api_key_env: Optional[str] = None

    @field_validator("options", mode="before")
    @classmethod
    def _options_default(cls, v):
        # An empty `options:` in YAML parses to None -> treat as no options.
        return v or {}


class ModelsConfig(BaseModel):
    strong: ModelConfig       # question generation + judge
    summarizer: ModelConfig   # the model under test
    answerer: ModelConfig     # constant across both conditions


class QuestionsConfig(BaseModel):
    count: int = 12
    types: list[str] = Field(default_factory=lambda: ["factual", "numeric", "inferential"])
    # What the generated questions target:
    #   "detailed" -> concrete facts, numbers, names, dates (fine-grained recall)
    #   "material" -> the key/material content a faithful summary must preserve
    #                 (obligations, rights, conditions, deadlines, liabilities),
    #                 rather than incidental trivia -> fairer for summary scoring
    focus: Literal["detailed", "material"] = "detailed"


class SummaryConfig(BaseModel):
    mode: Literal["generate", "file"] = "generate"
    target_words: int = 300


class AnsweringConfig(BaseModel):
    closed_book_baseline: bool = True
    require_evidence: bool = True
    max_context_chars: int = 24000
    # How much of the original is given to the answerer for the REFERENCE answer:
    #   "full"    -> the whole document (classic; may truncate long docs)
    #   "section" -> only the question's own section (+ preamble), with a fallback
    #                to the full document if the answer is not found there.
    # "section" requires section-tagged questions (sections.enabled) and avoids
    # truncation / saves context tokens. The candidate (summary) is unaffected.
    context_scope: Literal["full", "section"] = "full"


class JudgeConfig(BaseModel):
    scale: list[str] = Field(default_factory=lambda: ["match", "partial", "mismatch"])


class RunConfig(BaseModel):
    output_dir: str = "runs"


class SectionsConfig(BaseModel):
    # Generate questions per section (guaranteed coverage + per-section score)
    # instead of from the whole document at once.
    enabled: bool = False
    # Minimum number of detected headings for a heading strategy to be accepted;
    # below this the splitter falls through to the next strategy / paragraph
    # chunking.
    min_headings: int = 2
    # Target maximum size of a paragraph-chunk fallback section (characters).
    max_chunk_chars: int = 6000
    # Keep the text before the first heading as a "Preamble" section.
    keep_preamble: bool = True
    # Sections shorter than this are too small to yield good questions and are
    # skipped during question generation (they are still part of the document
    # that answers are read from).
    per_section_min_chars: int = 200


class Config(BaseModel):
    models: ModelsConfig
    questions: QuestionsConfig = Field(default_factory=QuestionsConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    answering: AnsweringConfig = Field(default_factory=AnsweringConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    sections: SectionsConfig = Field(default_factory=SectionsConfig)


def load_config(path: str | Path) -> Config:
    """Reads the YAML file and returns a validated Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(**data)
