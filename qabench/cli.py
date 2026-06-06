"""Command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from .config import load_config
from .pipeline.summarize import make_summary
from .providers import build_provider
from .report import compute_metrics, print_console, save_run
from .runner import load_or_generate_questions, run_benchmark

app = typer.Typer(add_completion=False, help="QA-Benchmark for summaries.")
console = Console()


def _apply_model_overrides(cfg, *, summarizer=None, answerer=None, strong=None) -> None:
    """Override model names per run without editing the config (options stay)."""
    if summarizer is not None:
        cfg.models.summarizer.model = summarizer
    if answerer is not None:
        cfg.models.answerer.model = answerer
    if strong is not None:
        cfg.models.strong.model = strong


@app.command()
def run(
    doc: Path = typer.Option(..., "--doc", "-d", help="Path to the original document."),
    config: Path = typer.Option("config.yaml", "--config", "-c", help="Configuration file."),
    summary: Optional[Path] = typer.Option(
        None, "--summary", "-s", help="Benchmark an existing summary (instead of generating)."
    ),
    regenerate_questions: bool = typer.Option(
        False, "--regenerate-questions", help="Ignore cached questions and regenerate."
    ),
    questions_file: Optional[Path] = typer.Option(
        None, "--questions", "-q",
        help="Reuse a fixed question set from a file (a previous run's questions.json / "
             "artifacts.json, or a cache file). Skips generation and the cache.",
    ),
    count: Optional[int] = typer.Option(
        None, "--count", "-n", help="Number of questions (overrides questions.count from the config)."
    ),
    summarizer: Optional[str] = typer.Option(
        None, "--summarizer", help="Override the summarizer model (the model under test)."
    ),
    answerer: Optional[str] = typer.Option(
        None, "--answerer", help="Override the answerer model."
    ),
    strong: Optional[str] = typer.Option(
        None, "--strong", help="Override the strong model (question generation + judge)."
    ),
    target_words: Optional[int] = typer.Option(
        None, "--target-words", "-w",
        help="Target summary length in words (overrides summary.target_words).",
    ),
):
    """Full benchmark run for a document."""
    cfg = load_config(config)
    if count is not None:
        cfg.questions.count = count
    if target_words is not None:
        cfg.summary.target_words = target_words
    _apply_model_overrides(cfg, summarizer=summarizer, answerer=answerer, strong=strong)

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting…", total=cfg.questions.count)

        def on_progress(done: int, total: int, desc: str) -> None:
            progress.update(task, completed=done, total=total, description=desc)

        result = run_benchmark(
            doc, cfg, summary_path=summary,
            regenerate_questions=regenerate_questions, questions_path=questions_file,
            on_progress=on_progress,
        )

    metrics = compute_metrics(result, cfg)
    print_console(result, cfg, metrics)
    out_dir = save_run(result, cfg, metrics)
    console.print(f"[bold]Report saved:[/bold] {out_dir}/report.md")


@app.command()
def questions(
    doc: Path = typer.Option(..., "--doc", "-d", help="Path to the original document."),
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    count: Optional[int] = typer.Option(
        None, "--count", "-n", help="Number of questions (overrides questions.count from the config)."
    ),
    strong: Optional[str] = typer.Option(
        None, "--strong", help="Override the strong model (question generation)."
    ),
    regenerate_questions: bool = typer.Option(
        False, "--regenerate-questions", help="Ignore cached questions and regenerate."
    ),
):
    """Generate (and cache) questions and display them.

    Uses the same cache as `run`, so a subsequent `run` with the same document
    and --count reuses exactly these questions.
    """
    from .language import detect_language
    from .loaders import load_document

    cfg = load_config(config)
    if count is not None:
        cfg.questions.count = count
    _apply_model_overrides(cfg, strong=strong)
    text = load_document(doc)[: cfg.answering.max_context_chars]
    language = detect_language(text)
    provider = build_provider(cfg.models.strong)
    qs = load_or_generate_questions(
        text, doc, cfg, provider, regenerate_questions, language
    )
    console.print(f"[dim]Language: {language}[/dim]")
    for q in qs:
        console.print(f"[bold]{q.id}.[/bold] ({q.type}) {q.text}")


@app.command()
def summarize(
    doc: Path = typer.Option(..., "--doc", "-d", help="Path to the original document."),
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Target file for the summary."),
    summarizer: Optional[str] = typer.Option(
        None, "--summarizer", help="Override the summarizer model."
    ),
    target_words: Optional[int] = typer.Option(
        None, "--target-words", "-w",
        help="Target summary length in words (overrides summary.target_words).",
    ),
):
    """Only create a summary with the summarizer model."""
    from .language import detect_language
    from .loaders import load_document

    cfg = load_config(config)
    if target_words is not None:
        cfg.summary.target_words = target_words
    _apply_model_overrides(cfg, summarizer=summarizer)
    text = load_document(doc)[: cfg.answering.max_context_chars]
    language = detect_language(text)
    provider = build_provider(cfg.models.summarizer)
    text_summary = make_summary(text, cfg, provider, language)
    if out:
        out.write_text(text_summary, encoding="utf-8")
        console.print(f"[bold]Saved:[/bold] {out}")
    else:
        console.print(text_summary)


if __name__ == "__main__":
    app()
