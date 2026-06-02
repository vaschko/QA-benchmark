"""Orchestrates the entire pipeline and caches generated questions.

Flow per question:
  reference = answer(question, CONTEXT=original)      # ground truth
  candidate = answer(question, CONTEXT=summary)
  baseline  = answer(question, NO context)            # contamination check
  judgement = judge(question, reference, candidate)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .language import detect_language
from .loaders import load_document
from .models import Question, QuestionResult, RunResult
from .pipeline.answer import answer_closed_book, answer_with_context
from .pipeline.judge import judge_answer
from .pipeline.questions import generate_questions
from .pipeline.summarize import make_summary
from .prompts import PROMPTS_VERSION
from .providers import build_provider

ProgressCb = Callable[[int, int, str], None]


def _questions_cache_path(text: str, doc_path: Path, cfg: Config, language: str) -> Path:
    key_src = (
        f"{text}|{cfg.questions.count}|{','.join(cfg.questions.types)}"
        f"|{language}|v{PROMPTS_VERSION}"
    )
    key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:10]
    cache_dir = Path(cfg.run.output_dir) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{doc_path.stem}_{key}.questions.json"


def load_or_generate_questions(
    text: str, doc_path: Path, cfg: Config, provider, regenerate: bool, language: str
) -> list[Question]:
    """Loads cached questions or generates them anew. Identical questions for the
    same document enable a fair comparison of multiple summaries."""
    cache_path = _questions_cache_path(text, doc_path, cfg, language)
    if cache_path.exists() and not regenerate:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return [Question(**q) for q in data]

    questions = generate_questions(text, cfg, provider, language)
    cache_path.write_text(
        json.dumps([q.model_dump() for q in questions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return questions


def run_benchmark(
    doc_path: str | Path,
    cfg: Config,
    summary_path: Optional[str | Path] = None,
    regenerate_questions: bool = False,
    on_progress: Optional[ProgressCb] = None,
) -> RunResult:
    doc_path = Path(doc_path)
    text = load_document(doc_path)

    truncated = False
    maxc = cfg.answering.max_context_chars
    if len(text) > maxc:
        text = text[:maxc]
        truncated = True

    # Detect the language once from the original and use it everywhere, so that
    # questions, summary and answers are produced in the document's language.
    language = detect_language(text)

    strong = build_provider(cfg.models.strong)
    answerer = build_provider(cfg.models.answerer)

    questions = load_or_generate_questions(
        text, doc_path, cfg, strong, regenerate_questions, language
    )

    # --- Summary: generate or load from file ---
    if summary_path is not None:
        summary = load_document(summary_path)
        summary_source = str(summary_path)
    else:
        summarizer = build_provider(cfg.models.summarizer)
        summary = make_summary(text, cfg, summarizer, language)
        summary_source = "generated"
    if len(summary) > maxc:
        summary = summary[:maxc]

    match_verdict = cfg.judge.scale[0]  # by convention the "full match" verdict
    total = len(questions)
    results: list[QuestionResult] = []

    for idx, q in enumerate(questions, start=1):
        if on_progress:
            on_progress(idx - 1, total, f"Question {idx}/{total}")

        reference = answer_with_context(q, text, cfg, answerer, language)
        candidate = answer_with_context(q, summary, cfg, answerer, language)
        baseline = (
            answer_closed_book(q, cfg, answerer, language)
            if cfg.answering.closed_book_baseline
            else None
        )

        valid = reference.found
        judgement = (
            judge_answer(q, reference, candidate, cfg, strong) if valid else None
        )
        baseline_judgement = (
            judge_answer(q, reference, baseline, cfg, strong)
            if (valid and baseline is not None and baseline.found)
            else None
        )
        discriminating = not (
            baseline_judgement is not None
            and baseline_judgement.verdict == match_verdict
        )

        results.append(
            QuestionResult(
                question=q,
                reference=reference,
                candidate=candidate,
                baseline=baseline,
                judgement=judgement,
                baseline_judgement=baseline_judgement,
                valid=valid,
                discriminating=discriminating,
            )
        )

    if on_progress:
        on_progress(total, total, "done")

    return RunResult(
        document_path=str(doc_path),
        document_chars=len(text),
        language=language,
        summary=summary,
        summary_source=summary_source,
        truncated=truncated,
        results=results,
    )
