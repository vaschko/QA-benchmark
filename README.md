# QA-Benchmark for summaries

Measures **how well a summary preserves the information of the original
document** – using local Ollama models ~~and/or the Claude API.~~

## The idea in one sentence

Questions are generated from the original and answered twice: once with the
**original** (= reference / ground truth) and once with the **summary** (=
candidate). A strong judge model compares the two. The **retention score** is
the share of questions that the summary lets you answer just as well as the
original.

## Why this holds up methodologically

- **Answerer model held constant:** original and summary are answered by the
  same `answerer` model – the only variable is the context.
- **No "memory":** every call is stateless (Ollama `/api/generate` without the
  `context` field), and the answerer is pinned to the document via the prompt
  (`NOT_IN_DOCUMENT`, mandatory evidence quote).
- **Contamination check:** every question is additionally answered **without**
  context (closed-book baseline). If the model can solve it from prior
  knowledge, the question is flagged as *contaminated* and excluded from the
  strict score.

## Language & number of questions

- **Language:** the document language is detected automatically (`langdetect`)
  and stated explicitly to the models – questions, summary and answers are
  always produced in the **language of the document** (English document →
  English questions).
- **Number of questions:** in [config.yaml](config.yaml) under
  `questions.count`, or per run with `--count / -n` (e.g. `-n 20`).
- **Question focus** (`questions.focus`, or `--focus` per run): `detailed`
  targets concrete facts, numbers and names (fine-grained recall); `material`
  targets the key content a faithful summary must preserve (obligations, rights,
  conditions, deadlines, liabilities) rather than incidental trivia. `material`
  is the fairer setting when scoring summaries, since a summary is *meant* to drop
  trivia. The two focuses are cached separately, so you can compare them:

  ```bash
  python -m qabench run -d doc.pdf --focus detailed
  python -m qabench run -d doc.pdf --focus material
  ```

## Three model roles (`config.yaml`)

| Role         | Task                                  | Recommendation              |
|--------------|---------------------------------------|-----------------------------|
| `strong`     | generate questions + judge answers    | strongest model / API       |
| `summarizer` | **the model under test**              | whatever you are benchmarking |
| `answerer`   | answer questions (constant!)          | mid-size, stable model      |

Each role has a `provider` field: `ollama` (local) or `anthropic` (Claude).
Switching is a single line.

## Installation

```bash
pip install -r requirements.txt   # PyMuPDF, python-docx, anthropic, langdetect
```

For the Claude API additionally:
```bash
export ANTHROPIC_API_KEY=sk-...
```

## Usage

> For a quick test there is `config.smoke.yaml` (small models, few questions):
> `python -m qabench run -d data/documents/sample_meridian.md -c config.smoke.yaml`

```bash
# Full run: generate a summary and benchmark it
python -m qabench run --doc data/documents/sample_meridian.md

# Override the number of questions per run (instead of questions.count in the config)
python -m qabench run --doc data/documents/sample_meridian.md --count 20

# Override model roles per run without editing the config
python -m qabench run --doc data/documents/sample_meridian.md --summarizer gemma4:e4b
#   available: --summarizer, --answerer, --strong (only the model name; options stay)

# Benchmark an already existing summary
python -m qabench run --doc data/documents/sample_meridian.md \
                      --summary data/summaries/my_summary.md

# Generate the questions, cache them and print them (no answering / scoring).
# A later `run` with the same document + --count reuses exactly these questions.
python -m qabench questions --doc data/documents/sample_meridian.md

# Only summarize
python -m qabench summarize --doc data/documents/sample_meridian.md -o out.md
```

### Comparing several summaries fairly

All comparisons below automatically run against the **same** set of questions
(see [Question caching](#question-caching) below).

**Comparing summarizer models** (which model writes the most informative
summary) – override `--summarizer` per run; questions and `answerer` stay
constant, so the only variable is the summarizer:

```bash
python -m qabench run -d doc.md --summarizer gemma4:e4b
python -m qabench run -d doc.md --summarizer qwen3.5:4b
python -m qabench run -d doc.md --summarizer mistral-small3.1
```

Then compare the **retention score** across the resulting `runs/<doc>_<ts>/`
reports (the `Models:` line records which summarizer was used).

**Comparing pre-built summaries** – pass a fixed summary file instead:

```bash
python -m qabench run -d doc.md -s summaries/method_A.md
python -m qabench run -d doc.md -s summaries/method_B.md
```

## Section-based benchmarking

The production summarizer works **section by section**, so the benchmark can too.
With `sections.enabled: true` (in [config.yaml](config.yaml)) the original is
first split into sections and questions are generated **per section** instead of
from the whole document at once. This gives two things:

- **Guaranteed coverage** – the question budget (`questions.count`) is
  distributed across sections proportional to their length, with at least one
  question per (non-trivial) section. Important clauses buried in the middle of a
  long T&C can no longer be skipped by the question generator.
- **A per-section retention score** – the report adds a *Retention by section*
  table, so you see *where* a summary loses information, not just an aggregate.

Splitting is deterministic and LLM-free. It tries several heading conventions in
order – Markdown (`#`), numbered (`1.` / `1.1` / `2)`), keyword (`Article` /
`Section` / `Clause`), roman numerals, ALL-CAPS headings – and falls back to
paragraph chunking, so a document is **always** split and nothing is lost.

Inspect how a document is split before running the benchmark (no LLM calls):

```bash
python -m qabench sections -d data/documents/Revolut_Terms.pdf
python -m qabench sections -d doc.md --show-content
```

Numbered headings may sit on their own line (`1.` with the title on the next
line, as many PDFs export) — these are detected too. `sections.max_depth` caps
how deep numbered/markdown headings split: `1` keeps only the top level
(`1.`, `2.`, …, not `1.1`), `2` adds one sublevel, `0` is unlimited. This stops
deeply numbered contracts from exploding into hundreds of micro-sections.

The clean original-vs-summary comparison and the contamination check are
preserved: the candidate is always answered against the summary, the baseline
without any context. Settings live under `sections:` in the config
(`min_headings`, `max_chunk_chars`, `keep_preamble`, `per_section_min_chars`,
`max_depth`).

### Section-scoped reference answering (`answering.context_scope`)

By default the **reference** answer (the ground truth from the original) is read
from the *whole* document (`context_scope: full`). For long documents that both
truncates (`max_context_chars`) and — with local Ollama, which has no prompt
caching — re-sends the full document on every answer call.

With `context_scope: section` the reference is instead read from the question's
**own section plus the preamble**, with a **fallback to the full document** when
the answer isn't found there (for clauses that depend on another section). This:

- **removes truncation** — each section easily fits the context window, so
  arbitrarily long documents become benchmarkable;
- **saves context tokens** — roughly 10–40× less context per reference call on
  Ollama;
- **improves the ground truth** — the answerer reads a focused section the
  question was generated from, instead of getting lost in a huge document.

It requires section-tagged questions (`sections.enabled: true`). The candidate
(summary) and baseline are unaffected.

## Question caching

Generating questions costs a call to the `strong` model, so questions are
**cached per document** under `runs/cache/`. The cache key is a hash of the
document text, `count`, question types, detected language and the prompt
version — it does **not** depend on the chosen models.

- The **first** `run` (or `questions`) for a given document + `--count`
  generates the questions and stores them — no flag needed.
- Every later `run` / `questions` with the same key **reuses** them
  automatically, instantly and without calling the model again. This is exactly
  what makes comparing different summarizers fair: they all face the identical
  questions.

### `--regenerate-questions`

This flag forces a **fresh** set of questions even when a cached one exists, and
overwrites the cache. Without it, an existing cache is always reused.

Use it when you want new questions, e.g.:
- after editing the document,
- to draw a different sample of questions,
- when starting a fresh benchmark series.

> When comparing several models, pass `--regenerate-questions` **only on the
> first run** so one canonical question set is created — then leave it off for
> all following runs so they reuse that exact set.

```bash
python -m qabench run -d doc.md --count 50 --summarizer model_A --regenerate-questions  # creates the set
python -m qabench run -d doc.md --count 50 --summarizer model_B                          # reuses it
python -m qabench run -d doc.md --count 50 --summarizer model_C                          # reuses it
```

### Reusing a fixed question set (`--questions`)

Every run also saves a standalone `questions.json` in its output folder. To run
against that exact set later — independent of document edits, `count` or prompt
changes — pass it explicitly:

```bash
python -m qabench run -d doc.md \
  --questions runs/doc_20260603_153007/questions.json \
  --summarizer model_X
```

`--questions` (short `-q`) skips generation and the cache entirely. It accepts a
run's `questions.json`, its `artifacts.json`, or a `runs/cache/*.questions.json`
file.

## Output

Each run produces `runs/<document>_<timestamp>/`:
- `report.md` – metrics + every question in detail (reference, candidate, evidence)
- `artifacts.json` – complete raw data (questions, answers, verdicts, config)
- `summary.txt` – the summary that was used
- `questions.json` – the question set used (reusable via `run --questions`)

In addition, an overview table is printed to the console.

### Metrics

Two underlying concepts first:

- **valid** – questions the **original** could answer. Questions the original
  itself can't answer are dropped (bad questions).
- **discriminating** – valid questions that the model can **not** solve from
  prior knowledge. A question is *contaminated* (not discriminating) when the
  closed-book answer (no document at all) already matched the reference — such a
  question says nothing about summary quality, the model knew it anyway.

The metrics:

- **Retention score** – of all **valid** questions, the share the summary lets
  you answer just as well as the original (judge verdict = `match`). The main
  quality number.
- **Retention score (discriminating only)** – the same ratio, but computed
  **only over discriminating questions** (contaminated ones excluded). The
  cleaner signal, because it ignores questions the model could have answered
  without any summary.
- **Coverage** – of all valid questions, the share for which the summary
  contained *any* answer at all (`found`), regardless of correctness. Coverage
  means "the summary said something relevant"; retention means "and it was
  right".
- **Contamination rate** – share of valid questions solvable from prior
  knowledge. Low = the benchmark is meaningful.

**Worked example** (from a real report):

```
Retention score: 22.4 % (11/49 valid questions)
Retention score (discriminating only): 21.4 %   # 9 of the 42 discriminating
Coverage: 28.6 %                                 # 14/49 had any answer
Contamination rate: 14.3 % (42/49 discriminating)  # 7 of 49 contaminated
```

Here 49 questions were valid; 7 were contaminated, leaving 42 discriminating.
The summary matched 11/49 questions (22.4 %), but only 9 of those were on
discriminating questions (9/42 = 21.4 %) — so 2 "hits" were freebies the model
knew anyway. It contained *some* answer for 14/49 (coverage), of which 11 were
correct.

> **Rule of thumb:** when the contamination rate is low, both retention scores
> are nearly equal (as above). When it is high, trust the **discriminating
> only** score.

## Project structure

```
qabench/
  config.py      configuration (pydantic)
  models.py      data models + JSON schemas
  loaders.py     txt / md / pdf / docx -> text
  splitter.py    robust section splitting (multi-strategy + fallback)
  prompts.py     all prompt templates (English)
  language.py    document language detection
  providers/     ollama | anthropic (interchangeable)
  pipeline/      questions / summarize / answer / judge
  runner.py      orchestration + question caching
  report.py      metrics + console + file reports
  cli.py         run / questions / sections / summarize
```
