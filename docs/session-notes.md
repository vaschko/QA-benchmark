# QA_Benchmark — Section-aware benchmarking (Änderungsübersicht)

_Stand: 2026-06-18 · Branch: `feature/section-splitter`_

## 1. Ausgangsproblem
Das Produktiv-Tool (Browser-Extension) fasst T&C **section-weise** zusammen, der
QA_Benchmark bewertete aber eine **monolithische** Zusammenfassung des
Gesamtdokuments. Der Benchmark maß damit eine andere Architektur als die
ausgespielte. Ziel: den Benchmark an die Section-für-Section-Pipeline angleichen
und robuster machen.

## 2. Die sechs Bausteine

**① Robuster Section-Splitter** (`qabench/splitter.py`, neu)
Der Splitter im Produkt war rein nummerierungsbasiert (`1.`/`1.1`) und lieferte
bei anderer Heading-Form 0 Sektionen. Neu: mehrere Strategien in Reihenfolge —
Markdown (`#`), nummeriert (`1.`, `1.1`, `2)`), Keyword (`Article`/`Section`/
`Clause`), römisch, GROSSBUCHSTABEN — plus Absatz-Fallback, damit nie etwas
verloren geht. Deterministisch, ohne LLM. Inspektion: `python -m qabench sections -d <doc>`.

**② Fragen pro Section + Pro-Section-Score** (`sections.enabled`)
Das Fragen-Budget (`questions.count`) wird längen-proportional über die Sektionen
verteilt (jede relevante Section ≥1 Frage) → garantierte Abdeckung. Jede Frage
ist mit ihrer Quell-Section getaggt; der Report bekommt eine
„Retention by section"-Tabelle → man sieht, *wo* Information verloren geht.

**③ Section-genaues Antworten** (`answering.context_scope: section`)
Die Referenzantwort (Ground Truth) wird nur gegen die eigene Section + Preamble
beantwortet, mit Fallback auf das Gesamtdokument. Effekt: keine Truncation mehr
(lange Docs benchmarkbar) und ~10–40× weniger Kontext-Tokens (Ollama hat kein
Caching).

**④ Frage-Fokus** (`questions.focus: detailed | material`)
`detailed` = konkrete Trivia (E-Mail, Firmennummer …). `material` = die
Kerninhalte, die eine treue Summary behalten muss (Pflichten, Rechte, Fristen,
Haftung). Analyse zeigte: 26 von 30 Fehlern waren „Summary schweigt zu Trivia" —
der Benchmark bestrafte das Weglassen von Trivia, was der Job einer Summary ist.
`material` misst fairer.

**⑤ Splitter-Bugfix für PDFs** (BMW)
Klauselnummern stehen in vielen PDFs allein auf einer Zeile (`1.`, Titel
darunter). Die Regex verlangte Text auf derselben Zeile → BMW kollabierte auf
3 Sektionen. Fix: Inhalt nach der Nummer optional, Titel aus der Folgezeile
(`1. General`). Neuer Schalter `sections.max_depth` (Default 1 = nur oberste
Ebene), damit tief nummerierte Verträge nicht in hunderte Mikro-Sektionen
zerfallen.

**⑥ Section-weise Zusammenfassung** (`summary.mode: per_section`)
Der Benchmark fasst optional jede Section einzeln zusammen und konkateniert —
spiegelt das Produkt. Adaptive Länge by default (keine starre Wortzahl); optional
`summary.compression` (Anteil je Section) — dokument-unabhängig, anders als die
absolute `-w`-Wortzahl.

## 3. Konkrete Ergebnisse

| Befund | Wert |
|---|---|
| BMW-Sektionen nach Splitter-Fix | 3 → 24 |
| Retention `detailed` vs `material` (gleiche ~2000-W-Summary) | 30,6 % → 51,0 % |
| Coverage dabei | 46,9 % → 75,5 % |
| Fehler-Diagnose (detailed-Lauf) | 26/30 Mismatches = Summary schweigt zu Trivia (kein Bug) |

## 4. Neue Konfigurationsschalter

| Schalter | Bedeutung | Default |
|---|---|---|
| `sections.enabled` | Fragen pro Section + Pro-Section-Score | `true` |
| `sections.max_depth` | Split-Tiefe (1 = top-level) | `1` |
| `answering.context_scope` | Referenz gegen `full` oder `section` | `section` |
| `questions.focus` | `detailed` (Trivia) / `material` (Kern) | `material` |
| `summary.mode` | `generate` / `per_section` / `file` | `per_section` |
| `summary.compression` | optionale Rate statt fester Wortzahl | aus (adaptiv) |

`python -m qabench run -d <doc>` macht jetzt ohne Flags: per_section-Summary +
material-Fragen + section-scoped answering.

## 5. Methodische Kernpunkte
- Saubere Methodik blieb erhalten: answerer-Modell konstant, Kandidat-vs-Referenz,
  Kontaminationscheck (closed-book). Sektioniert wurden nur Fragegenerierung,
  Referenz-Scope und Summary-Erzeugung.
- Retention immer neben der Compression-% lesen → fairer Modell-Vergleich auch bei
  ungleichen Längen; absolute Wortzahlen sind dokumentabhängig.
- Splitting ist regelbasiert (kein LLM) → reproduzierbar.
