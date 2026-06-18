"""Stage 1: generate questions from the original document (strong model).

Two modes:
  * whole document  -- `generate_questions` (one call, the original behavior)
  * per section      -- `generate_questions_by_section` (one call per section,
    the question budget distributed across sections so every section is probed)
"""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..jsonutil import parse_json
from ..models import Question, questions_schema
from ..providers import LLMProvider
from ..splitter import Section


def _parse_items(raw: str) -> list[dict]:
    data = parse_json(raw)
    return data.get("questions", []) if isinstance(data, dict) else data


def generate_questions(
    text: str, cfg: Config, provider: LLMProvider, language: str
) -> list[Question]:
    system, user = prompts.question_gen(
        cfg.questions.count, cfg.questions.types, language, cfg.questions.focus
    )
    schema = questions_schema(cfg.questions.types)

    raw = provider.generate(system=system, prompt=user, json_schema=schema, cacheable=text)
    items = _parse_items(raw)
    questions: list[Question] = []
    for i, item in enumerate(items[: cfg.questions.count], start=1):
        questions.append(
            Question(id=i, text=item["text"], type=item.get("type", "factual"))
        )
    if not questions:
        raise RuntimeError("No questions could be generated.")
    return questions


def allocate_questions(
    sections: list[Section], total: int, min_chars: int
) -> list[tuple[Section, int]]:
    """Distribute `total` questions across sections proportional to their length.

    Every eligible section (>= `min_chars`) gets at least one question, so the
    whole document is covered. When there are more eligible sections than the
    budget allows, the `total` longest sections each get one. Returns
    (section, k) pairs in document order.
    """
    candidates = [s for s in sections if len(s.content.strip()) >= min_chars] or list(sections)
    n = len(candidates)
    if total <= 0 or n == 0:
        return []

    if n >= total:
        # Not enough budget for one each -> give one to the `total` longest.
        longest = sorted(range(n), key=lambda i: len(candidates[i].content), reverse=True)[:total]
        return [(candidates[i], 1) for i in sorted(longest)]

    # One each, then distribute the remainder by length (largest-remainder method).
    alloc = [1] * n
    remaining = total - n
    lengths = [len(candidates[i].content) for i in range(n)]
    total_len = sum(lengths) or 1
    ideal = [remaining * length / total_len for length in lengths]
    base = [int(x) for x in ideal]
    for i in range(n):
        alloc[i] += base[i]
    leftover = remaining - sum(base)
    for i in sorted(range(n), key=lambda i: ideal[i] - base[i], reverse=True)[:leftover]:
        alloc[i] += 1
    return [(candidates[i], alloc[i]) for i in range(n)]


def generate_questions_by_section(
    sections: list[Section], cfg: Config, provider: LLMProvider, language: str
) -> list[Question]:
    """Generate questions section by section (one strong-model call per section)."""
    schema = questions_schema(cfg.questions.types)
    allocation = allocate_questions(
        sections, cfg.questions.count, cfg.sections.per_section_min_chars
    )
    questions: list[Question] = []
    qid = 1
    for section, k in allocation:
        system, user = prompts.question_gen_section(
            k, cfg.questions.types, language, section.title, cfg.questions.focus
        )
        raw = provider.generate(
            system=system, prompt=user, json_schema=schema, cacheable=section.content
        )
        for item in _parse_items(raw)[:k]:
            questions.append(
                Question(
                    id=qid,
                    text=item["text"],
                    type=item.get("type", "factual"),
                    section_index=section.index,
                    section_title=section.title,
                )
            )
            qid += 1
    if not questions:
        raise RuntimeError("No questions could be generated from any section.")
    return questions
