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
    focus: Optional[str] = typer.Option(
        None, "--focus",
        help="Question focus: 'detailed' (concrete trivia) or 'material' (key "
             "content a summary should preserve). Overrides questions.focus.",
    ),
    summary_mode: Optional[str] = typer.Option(
        None, "--summary-mode",
        help="Summary mode: 'generate' (one global summary) or 'per_section' "
             "(summarize each section). Overrides summary.mode.",
    ),
):
    """Full benchmark run for a document."""
    cfg = load_config(config)
    if count is not None:
        cfg.questions.count = count
    if target_words is not None:
        cfg.summary.target_words = target_words
    if focus is not None:
        cfg.questions.focus = focus
    if summary_mode is not None:
        cfg.summary.mode = summary_mode
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
    focus: Optional[str] = typer.Option(
        None, "--focus", help="Question focus: 'detailed' or 'material' (overrides questions.focus)."
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
    if focus is not None:
        cfg.questions.focus = focus
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
def sections(
    doc: Path = typer.Option(..., "--doc", "-d", help="Path to the document."),
    config: Path = typer.Option("config.yaml", "--config", "-c"),
    show_content: bool = typer.Option(
        False, "--show-content", help="Also print the start of each section's text."
    ),
):
    """Split a document into sections and show the result (no LLM calls).

    Use this to check how a document is divided before benchmarking -- it makes
    the chosen strategy (markdown / numbered / keyword / roman / allcaps /
    paragraph fallback) and the section boundaries visible.
    """
    from .loaders import load_document
    from .splitter import split_into_sections

    cfg = load_config(config)
    text = load_document(doc)
    secs = split_into_sections(
        text,
        min_headings=cfg.sections.min_headings,
        max_chunk_chars=cfg.sections.max_chunk_chars,
        keep_preamble=cfg.sections.keep_preamble,
        max_depth=cfg.sections.max_depth,
    )
    method = secs[0].method if secs else "-"
    console.print(
        f"[dim]{len(secs)} sections via '{method}' strategy "
        f"({len(text)} chars total)[/dim]\n"
    )
    for s in secs:
        console.print(
            f"[bold]{s.index}.[/bold] [cyan]{s.title}[/cyan]  "
            f"[dim](L{s.level}, {s.char_count} chars)[/dim]"
        )
        if show_content:
            preview = s.content[:400].replace("\n", " ")
            console.print(f"   [dim]{preview}{'…' if s.char_count > 400 else ''}[/dim]\n")


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
    summary_mode: Optional[str] = typer.Option(
        None, "--summary-mode",
        help="Summary mode: 'generate' or 'per_section' (overrides summary.mode).",
    ),
):
    """Only create a summary with the summarizer model."""
    from .language import detect_language
    from .loaders import load_document
    from .pipeline.summarize import make_section_summaries

    cfg = load_config(config)
    if target_words is not None:
        cfg.summary.target_words = target_words
    if summary_mode is not None:
        cfg.summary.mode = summary_mode
    _apply_model_overrides(cfg, summarizer=summarizer)
    text = load_document(doc)[: cfg.answering.max_context_chars]
    language = detect_language(text)
    provider = build_provider(cfg.models.summarizer)
    if cfg.summary.mode == "per_section":
        text_summary = make_section_summaries(text, cfg, provider, language)
    else:
        text_summary = make_summary(text, cfg, provider, language)
    if out:
        out.write_text(text_summary, encoding="utf-8")
        console.print(f"[bold]Saved:[/bold] {out}")
    else:
        console.print(text_summary)


if __name__ == "__main__":
    app()
