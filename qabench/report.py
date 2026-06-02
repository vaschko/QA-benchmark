"""Result aggregation, console table and file reports."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import Config
from .models import RunResult


def compute_metrics(result: RunResult, cfg: Config) -> dict[str, Any]:
    match_verdict = cfg.judge.scale[0]
    results = result.results

    valid = [r for r in results if r.valid]
    n_valid = len(valid)
    discriminating = [r for r in valid if r.discriminating]

    def is_match(r) -> bool:
        return r.judgement is not None and r.judgement.verdict == match_verdict

    matches = [r for r in valid if is_match(r)]
    matches_discr = [r for r in discriminating if is_match(r)]

    def pct(part: int, whole: int) -> float:
        return round(100 * part / whole, 1) if whole else 0.0

    # Verdict distribution
    verdict_counts: dict[str, int] = {v: 0 for v in cfg.judge.scale}
    for r in valid:
        if r.judgement:
            verdict_counts[r.judgement.verdict] = verdict_counts.get(r.judgement.verdict, 0) + 1

    # Breakdown by question type
    by_type: dict[str, dict[str, int]] = {}
    for r in valid:
        t = r.question.type
        bucket = by_type.setdefault(t, {"total": 0, "match": 0})
        bucket["total"] += 1
        if is_match(r):
            bucket["match"] += 1

    return {
        "total_questions": len(results),
        "valid_questions": n_valid,
        "invalid_questions": len(results) - n_valid,
        "discriminating_questions": len(discriminating),
        "retention_score": pct(len(matches), n_valid),
        "retention_score_strict": pct(len(matches_discr), len(discriminating)),
        "coverage": pct(sum(1 for r in valid if r.candidate.found), n_valid),
        "contamination_rate": pct(n_valid - len(discriminating), n_valid),
        "verdict_counts": verdict_counts,
        "by_type": by_type,
        "match_verdict": match_verdict,
    }


def print_console(result: RunResult, cfg: Config, metrics: dict[str, Any]) -> None:
    console = Console()

    console.print()
    console.rule("[bold]QA-Benchmark Result")
    console.print(f"Document: {result.document_path}")
    console.print(f"Language: {result.language}")
    console.print(f"Summary:  {result.summary_source}")
    if result.truncated:
        console.print("[yellow]⚠ Document was truncated to max_context_chars.[/yellow]")
    console.print()

    console.print(f"[bold green]Retention score: {metrics['retention_score']} %[/bold green]  "
                  f"(discriminating questions only: {metrics['retention_score_strict']} %)")
    console.print(f"Coverage (summary could answer): {metrics['coverage']} %")
    console.print(f"Contamination rate (prior knowledge): {metrics['contamination_rate']} %  "
                  f"-> {metrics['discriminating_questions']}/{metrics['valid_questions']} questions discriminating")
    console.print()

    table = Table(show_lines=False, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Type")
    table.add_column("Question", max_width=50)
    table.add_column("Ref")
    table.add_column("Verdict")
    table.add_column("Baseline")
    table.add_column("Flags")

    for r in result.results:
        verdict = r.judgement.verdict if r.judgement else "—"
        color = {"match": "green", "partial": "yellow"}.get(verdict, "red")
        ref_state = "✓" if r.reference.found else "[red]∅[/red]"
        base_state = "knew" if (r.baseline and r.baseline.found) else "—"
        flags = []
        if not r.valid:
            flags.append("[red]invalid[/red]")
        if not r.discriminating:
            flags.append("[magenta]contaminated[/magenta]")
        table.add_row(
            str(r.question.id),
            r.question.type,
            r.question.text,
            ref_state,
            f"[{color}]{verdict}[/{color}]",
            base_state,
            " ".join(flags),
        )
    console.print(table)
    console.print()


def save_run(result: RunResult, cfg: Config, metrics: dict[str, Any]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(result.document_path).stem
    out_dir = Path(cfg.run.output_dir) / f"{stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "timestamp": ts,
        "config": cfg.model_dump(),
        "metrics": metrics,
        "result": result.model_dump(),
    }
    (out_dir / "artifacts.json").write_text(
        json.dumps(artifacts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(_render_markdown(result, cfg, metrics), encoding="utf-8")
    (out_dir / "summary.txt").write_text(result.summary, encoding="utf-8")
    return out_dir


def _render_markdown(result: RunResult, cfg: Config, m: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# QA-Benchmark Report\n")
    lines.append(f"- **Document:** `{result.document_path}` ({result.document_chars} chars)")
    lines.append(f"- **Language:** {result.language}")
    lines.append(f"- **Summary:** {result.summary_source}")
    lines.append(f"- **Models:** strong=`{cfg.models.strong.model}`, "
                 f"summarizer=`{cfg.models.summarizer.model}`, "
                 f"answerer=`{cfg.models.answerer.model}`")
    if result.truncated:
        lines.append("- ⚠ **Document truncated** to `max_context_chars`.")
    lines.append("")
    lines.append("## Metrics\n")
    lines.append(f"- **Retention score:** {m['retention_score']} % "
                 f"({m['verdict_counts'].get(m['match_verdict'], 0)}/{m['valid_questions']} valid questions)")
    lines.append(f"- **Retention score (discriminating only):** {m['retention_score_strict']} %")
    lines.append(f"- **Coverage:** {m['coverage']} %")
    lines.append(f"- **Contamination rate:** {m['contamination_rate']} % "
                 f"({m['discriminating_questions']}/{m['valid_questions']} discriminating)")
    lines.append(f"- **Verdicts:** " + ", ".join(f"{k}={v}" for k, v in m["verdict_counts"].items()))
    lines.append("")
    if m["by_type"]:
        lines.append("### By question type\n")
        lines.append("| Type | Matches | Total |")
        lines.append("|---|---|---|")
        for t, b in m["by_type"].items():
            lines.append(f"| {t} | {b['match']} | {b['total']} |")
        lines.append("")

    lines.append("## Questions in detail\n")
    for r in result.results:
        verdict = r.judgement.verdict if r.judgement else "—"
        lines.append(f"### {r.question.id}. ({r.question.type}) {r.question.text}\n")
        lines.append(f"- **Verdict:** {verdict}" + (f" — {r.judgement.reason}" if r.judgement else ""))
        lines.append(f"- **Reference (original):** {r.reference.answer}")
        if r.reference.evidence:
            lines.append(f"  - Evidence: _{r.reference.evidence}_")
        lines.append(f"- **Candidate (summary):** {r.candidate.answer}")
        if r.candidate.evidence:
            lines.append(f"  - Evidence: _{r.candidate.evidence}_")
        if r.baseline is not None:
            lines.append(f"- **Baseline (no context):** {r.baseline.answer}")
        flags = []
        if not r.valid:
            flags.append("invalid (original could not answer)")
        if not r.discriminating:
            flags.append("contaminated (solvable from prior knowledge)")
        if flags:
            lines.append(f"- **Flags:** {', '.join(flags)}")
        lines.append("")
    return "\n".join(lines)
