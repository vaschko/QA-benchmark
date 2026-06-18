"""Robust, deterministic splitting of a document into sections.

The production tool (SummarizationBenchmark) splits T&C strictly on the
numbering convention ``1.`` / ``1.1``. That silently drops every document that
uses a different heading style -- Markdown headers, ``Article N``, ALL-CAPS
headings, roman numerals -- and produces *zero* sections for unstructured text.
For a benchmark that must run across heterogeneous documents that is too
fragile.

This module tries several heading conventions in priority order and, if none
fits, falls back to paragraph chunking. A document is therefore *always* split
into usable sections and nothing is ever lost. Splitting is purely rule-based
(no LLM), so it is deterministic and reproducible -- a property a benchmark
needs.

Strategy order (first one that yields >= ``min_headings`` headings wins):
    1. markdown   ATX headers (``#`` .. ``######``)
    2. numbered   ``1.`` / ``1.1`` / ``1.2.3`` / ``2)`` / ``3|``
    3. keyword    ``Article 1`` / ``Section IV`` / ``Clause A`` / ``Part 2`` ...
    4. roman      ``I.`` / ``II)`` ...
    5. allcaps    short ALL-CAPS heading lines (``PRIVACY POLICY``)
    fallback      greedy paragraph chunking under ``max_chunk_chars``
"""

from __future__ import annotations

import re

from pydantic import BaseModel


class Section(BaseModel):
    index: int          # 0-based position in the document
    title: str          # heading text, or a synthesized label ("Preamble", "Part 2")
    content: str        # full section text, INCLUDING the heading line
    level: int = 1      # heading depth where known (markdown #, decimal dots)
    method: str = ""    # which strategy produced the split (for transparency)

    @property
    def char_count(self) -> int:
        return len(self.content)


# --------------------------------------------------------------------------
# Heading patterns -- each matches a *heading line* anchored at line start.
# --------------------------------------------------------------------------
_MD_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+\S.*$", re.MULTILINE)

# A single integer must carry a separator (``1.`` / ``2)`` / ``3|``); a bare
# integer would also match prose like "2024 was ..." . Multi-level numbers
# (``1.1`` / ``1.2.3``) are accepted without a trailing separator.
_NUM_HEADING = re.compile(
    r"^[ \t]*(?:\d+\.\d+(?:\.\d+)*|\d+[.)|])[ \t]+\S.*$", re.MULTILINE
)

_KEYWORD_HEADING = re.compile(
    r"^[ \t]*(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause|CHAPTER|Chapter|PART|Part)"
    r"[ \t]+(?:\d+|[IVXLCDM]+|[A-Z])\b.*$",
    re.MULTILINE,
)

_ROMAN_HEADING = re.compile(r"^[ \t]*[IVXLCDM]{1,7}[.)][ \t]+\S.*$", re.MULTILINE)

# Whole line is upper-case letters/digits/light punctuation (>= 4 chars), no
# trailing sentence period. Heuristic, hence lowest priority.
_ALLCAPS_HEADING = re.compile(r"^[ \t]*[A-Z][A-Z0-9 ,'&/\-]{2,58}[A-Z0-9]$", re.MULTILINE)


def _md_level(line: str) -> int:
    m = re.match(r"^[ \t]*(#{1,6})", line)
    return len(m.group(1)) if m else 1


def _num_level(line: str) -> int:
    m = re.match(r"^[ \t]*(\d+(?:\.\d+)*)", line)
    return m.group(1).count(".") + 1 if m else 1


def _const_level(_line: str) -> int:
    return 1


def _clean_title(line: str, *, strip_md: bool = False) -> str:
    t = line.strip()
    if strip_md:
        t = re.sub(r"^#{1,6}[ \t]*", "", t)
    return t or "Untitled"


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------
def _sectionize(text, matches, method, level_fn, keep_preamble) -> list[Section]:
    """Turn heading matches into Sections, optionally keeping the preamble."""
    out: list[Section] = []
    idx = 0

    preamble = text[: matches[0].start()].strip()
    if keep_preamble and preamble:
        out.append(Section(index=idx, title="Preamble", content=preamble, method=method))
        idx += 1

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[m.start() : end].strip()
        if not block:
            continue
        heading_line = block.splitlines()[0].strip()
        out.append(
            Section(
                index=idx,
                title=_clean_title(heading_line, strip_md=(method == "markdown")),
                content=block,
                level=level_fn(heading_line),
                method=method,
            )
        )
        idx += 1
    return out


def _split_paragraphs(text: str, max_chunk_chars: int) -> list[Section]:
    """Fallback: greedily pack paragraphs into chunks under ``max_chunk_chars``.

    Guarantees the whole document is covered even when no heading style is
    detected. An oversized single paragraph is hard-sliced as a last resort.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    cur = ""
    for para in re.split(r"\n[ \t]*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) > max_chunk_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(para), max_chunk_chars):
                chunks.append(para[i : i + max_chunk_chars])
            continue
        if cur and len(cur) + len(para) + 2 > max_chunk_chars:
            chunks.append(cur)
            cur = para
        else:
            cur = f"{cur}\n\n{para}" if cur else para
    if cur:
        chunks.append(cur)

    return [
        Section(index=i, title=f"Part {i + 1}", content=c, method="paragraph")
        for i, c in enumerate(chunks)
    ]


def split_into_sections(
    text: str,
    *,
    min_headings: int = 2,
    max_chunk_chars: int = 6000,
    keep_preamble: bool = True,
) -> list[Section]:
    """Split ``text`` into sections, robust across heading conventions.

    Tries each heading strategy in priority order; the first that finds at least
    ``min_headings`` headings wins. If none does, falls back to paragraph
    chunking so nothing is ever lost.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    strategies = (
        (_MD_HEADING, "markdown", _md_level),
        (_NUM_HEADING, "numbered", _num_level),
        (_KEYWORD_HEADING, "keyword", _const_level),
        (_ROMAN_HEADING, "roman", _const_level),
        (_ALLCAPS_HEADING, "allcaps", _const_level),
    )

    for regex, method, level_fn in strategies:
        matches = list(regex.finditer(text))
        if len(matches) >= min_headings:
            return _sectionize(text, matches, method, level_fn, keep_preamble)

    return _split_paragraphs(text, max_chunk_chars)
