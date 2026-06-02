"""Stage 4: judge the candidate answer against the reference answer (strong model)."""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..jsonutil import parse_json
from ..models import Answer, Judgement, Question, judgement_schema
from ..providers import LLMProvider


def judge_answer(
    question: Question,
    reference: Answer,
    candidate: Answer,
    cfg: Config,
    provider: LLMProvider,
) -> Judgement:
    system, user_tpl = prompts.judge(cfg.judge.scale)
    user = user_tpl.format(
        question=question.text,
        reference=reference.answer,
        candidate=candidate.answer,
    )
    schema = judgement_schema(cfg.judge.scale)
    raw = provider.generate(system=system, prompt=user, json_schema=schema)
    data = parse_json(raw)

    verdict = str(data.get("verdict", "")).strip()
    if verdict not in cfg.judge.scale:
        # Fall back to the "worst" verdict if the model goes off the rails.
        verdict = cfg.judge.scale[-1]
    return Judgement(verdict=verdict, reason=str(data.get("reason", "")))
