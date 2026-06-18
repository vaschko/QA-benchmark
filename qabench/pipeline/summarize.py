"""Stage 2: create the summary (the model under test).

Two modes:
  * generate    -- one summary over the whole document (`make_summary`)
  * per_section -- summarize each section separately and concatenate
    (`make_section_summaries`), mirroring a section-by-section production
    summarizer. Section length stays adaptive unless a compression ratio is set.
"""

from __future__ import annotations

from .. import prompts
from ..config import Config
from ..providers import LLMProvider
from ..splitter import split_into_sections


def make_summary(text: str, cfg: Config, provider: LLMProvider, language: str) -> str:
    system, user = prompts.summarize(cfg.summary.target_words, language)
    raw = provider.generate(system=system, prompt=user, cacheable=text)
    return raw.strip()


def make_section_summaries(
    text: str, cfg: Config, provider: LLMProvider, language: str
) -> str:
    """Summarize the document section by section and concatenate the parts.

    Uses the same splitter (and settings) as question generation, so the summary
    is built over the same sections that are scored. Each section summary is
    prefixed with its title, giving a structured summary that mirrors the
    production tool's per-section output.
    """
    sections = split_into_sections(
        text,
        min_headings=cfg.sections.min_headings,
        max_chunk_chars=cfg.sections.max_chunk_chars,
        keep_preamble=cfg.sections.keep_preamble,
        max_depth=cfg.sections.max_depth,
    )
    ratio = cfg.summary.compression
    min_chars = cfg.sections.per_section_min_chars

    parts: list[str] = []
    for s in sections:
        content = s.content.strip()
        if not content:
            continue
        # Too short to be worth a summarization call -- keep verbatim (the content
        # already begins with its own heading, so no extra title prefix).
        if len(content) < min_chars:
            parts.append(content)
            continue
        target = max(20, round(len(content.split()) * ratio)) if ratio else None
        system, user = prompts.summarize_section(target, language)
        raw = provider.generate(system=system, prompt=user, cacheable=content)
        parts.append(f"## {s.title}\n{raw.strip()}")

    return "\n\n".join(parts).strip()
