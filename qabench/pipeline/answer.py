"""Stage 3: answer a question -- grounded (with context) or closed-book.

Important: the same provider (the `answerer` model) is used for the original and
the summary, so that the only variable is the context.
"""

from __future__ import annotations

from typing import Optional

from .. import prompts
from ..config import Config
from ..jsonutil import parse_json
from ..models import ANSWER_SCHEMA, Answer, Question
from ..providers import LLMProvider


def answer_with_context(
    question: Question, context: str, cfg: Config, provider: LLMProvider, language: str
) -> Answer:
    """Answers the question exclusively from the given context."""
    system, user_tpl = prompts.answer_grounded(cfg.answering.require_evidence, language)
    user = user_tpl.format(question=question.text)
    raw = provider.generate(
        system=system, prompt=user, json_schema=ANSWER_SCHEMA, cacheable=context
    )
    return _to_answer(raw)


def answer_closed_book(
    question: Question, cfg: Config, provider: LLMProvider, language: str
) -> Answer:
    """Answers the question without any context (contamination check)."""
    system, user_tpl = prompts.answer_closed_book(language)
    user = user_tpl.format(question=question.text)
    raw = provider.generate(system=system, prompt=user, json_schema=ANSWER_SCHEMA)
    return _to_answer(raw)


def _to_answer(raw: str) -> Answer:
    data = parse_json(raw)
    return Answer(
        found=bool(data.get("found", False)),
        answer=str(data.get("answer", "")),
        evidence=str(data.get("evidence", "")),
    )
