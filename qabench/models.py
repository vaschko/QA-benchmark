"""Data models (pydantic) for internal use and serialization, plus flat JSON
schemas for the structured model output.

We deliberately use hand-written, *flat* schemas for the LLM input/output
(instead of pydantic.model_json_schema()), because both Ollama's `format` and
Anthropic's tool-use work most reliably with simple schemas without $ref/$defs.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# --------------------------------------------------------------------------
# Internal data models
# --------------------------------------------------------------------------
class Question(BaseModel):
    id: int
    text: str
    type: str = "factual"


class Answer(BaseModel):
    found: bool          # was the question answered from the given context?
    answer: str          # the answer itself (or NOT_IN_DOCUMENT / NOT_KNOWN)
    evidence: str = ""   # verbatim supporting quote from the context


class Judgement(BaseModel):
    verdict: str         # one of the values from config.judge.scale
    reason: str = ""


class QuestionResult(BaseModel):
    question: Question
    reference: Answer                      # answer from the original (ground truth)
    candidate: Answer                      # answer from the summary
    baseline: Optional[Answer] = None      # closed-book (no context)
    judgement: Optional[Judgement] = None  # candidate vs. reference
    baseline_judgement: Optional[Judgement] = None  # baseline vs. reference
    valid: bool = True                     # reference could be answered
    discriminating: bool = True            # not solvable from prior knowledge


class RunResult(BaseModel):
    document_path: str
    document_chars: int
    document_words: int = 0      # word count of the (truncated) document
    language: str = ""           # detected document language
    summary: str
    summary_source: str          # "generated" or path to the summary file
    truncated: bool = False
    results: list[QuestionResult] = []


# --------------------------------------------------------------------------
# Flat JSON schemas for structured model outputs
# --------------------------------------------------------------------------
def questions_schema(types: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": types},
                    },
                    "required": ["text", "type"],
                },
            }
        },
        "required": ["questions"],
    }


ANSWER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["found", "answer", "evidence"],
}


def judgement_schema(scale: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": scale},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
    }
