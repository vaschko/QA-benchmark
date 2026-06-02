"""Stage 1: generate questions from the original document (strong model)."""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..jsonutil import parse_json
from ..models import Question, questions_schema
from ..providers import LLMProvider


def generate_questions(
    text: str, cfg: Config, provider: LLMProvider, language: str
) -> list[Question]:
    system, user = prompts.question_gen(cfg.questions.count, cfg.questions.types, language)
    schema = questions_schema(cfg.questions.types)

    raw = provider.generate(system=system, prompt=user, json_schema=schema, cacheable=text)
    data = parse_json(raw)

    items = data.get("questions", []) if isinstance(data, dict) else data
    questions: list[Question] = []
    for i, item in enumerate(items[: cfg.questions.count], start=1):
        questions.append(
            Question(id=i, text=item["text"], type=item.get("type", "factual"))
        )
    if not questions:
        raise RuntimeError("No questions could be generated.")
    return questions
