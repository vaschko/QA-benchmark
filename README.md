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
pip install -r requirements.txt   # pypdf, python-docx, anthropic, langdetect
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

## Output

Each run produces `runs/<document>_<timestamp>/`:
- `report.md` – metrics + every question in detail (reference, candidate, evidence)
- `artifacts.json` – complete raw data (questions, answers, verdicts, config)
- `summary.txt` – the summary that was used

In addition, an overview table is printed to the console.

### Metrics

- **Retention score** – share of questions with a full match
  (candidate ≈ reference).
- **Retention score (strict)** – the same, but only over discriminating
  (non-contaminated) questions.
- **Coverage** – share of questions the summary could answer at all.
- **Contamination rate** – share of questions the model solved from prior knowledge.

## Project structure

```
qabench/
  config.py      configuration (pydantic)
  models.py      data models + JSON schemas
  loaders.py     txt / md / pdf / docx -> text
  prompts.py     all prompt templates (English)
  language.py    document language detection
  providers/     ollama | anthropic (interchangeable)
  pipeline/      questions / summarize / answer / judge
  runner.py      orchestration + question caching
  report.py      metrics + console + file reports
  cli.py         run / questions / summarize
```
