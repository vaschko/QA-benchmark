"""Central, versioned prompt templates (English).

All prompts in one place so the methodology is easy to revise and audit. The
grounding prompts are intentionally strict in order to suppress the model's
parametric prior knowledge.
"""

from __future__ import annotations
import textwrap

PROMPTS_VERSION = "5"

NOT_IN_DOC = "NOT_IN_DOCUMENT"
NOT_KNOWN = "NOT_KNOWN"


# --------------------------------------------------------------------------
# Question generation (strong model)
# --------------------------------------------------------------------------
def _exam_designer_system(focus: str, scope: str) -> str:
    """Build the exam-designer system prompt.

    `scope` is "document" or "section"; `focus` is "detailed" (concrete
    facts/numbers/names) or "material" (the key content a faithful summary must
    preserve, rather than incidental trivia).
    """
    if scope == "section":
        intro = (
            "You are an exam designer. From a given section of a larger document "
            "you create questions that can be answered exclusively from the "
            "content of that section. "
        )
    else:
        intro = (
            "You are an exam designer. From a given document you create questions "
            "that can be answered exclusively from the content of that document. "
        )

    if focus == "material":
        body = (
            f"Good questions target the MATERIAL content that a faithful summary "
            f"must preserve: the key obligations, rights, permissions, "
            f"prohibitions, conditions, deadlines, liabilities, remedies and the "
            f"main point of each provision in the {scope}. Prefer what changes "
            f"the parties' rights or duties over incidental trivia such as exact "
            f"email addresses, individual entity names or minor cross-references. "
        )
    else:
        body = (
            f"Good questions are specific and target concrete facts, numbers, "
            f"names, dates, definitions or relationships in the {scope}. "
        )

    outro = (
        f"Avoid questions that anyone could answer from general knowledge without "
        f"the {scope}, and questions whose answer is not in the {scope}."
    )
    return intro + body + outro


def question_gen(
    count: int, types: list[str], language: str, focus: str = "detailed"
) -> tuple[str, str]:
    system = _exam_designer_system(focus, "document")
    user = (
        f"Create exactly {count} questions about the following document. "
        f"Spread the questions across these types where possible: {', '.join(types)}. "
        "Cover different parts of the document. "
        f"IMPORTANT: Language of the questions: {language}. Write ALL questions "
        "exclusively in this language, regardless of the language of this instruction. "
        "Return the result exclusively as JSON according to the schema."
    )
    return system, user


def question_gen_section(
    count: int, types: list[str], language: str, section_title: str, focus: str = "detailed"
) -> tuple[str, str]:
    """Question generation from a single SECTION of a larger document.

    Same exam-designer rules as `question_gen`, but the model is told the text is
    one section so it targets that section instead of a whole document.
    """
    system = _exam_designer_system(focus, "section")
    user = (
        f"Create exactly {count} question(s) about the following section "
        f'("{section_title}") of a larger document. '
        f"Spread the questions across these types where possible: {', '.join(types)}. "
        "Every question must be answerable from THIS section's content alone. "
        f"IMPORTANT: Language of the questions: {language}. Write ALL questions "
        "exclusively in this language, regardless of the language of this instruction. "
        "Return the result exclusively as JSON according to the schema."
    )
    return system, user


# --------------------------------------------------------------------------
# Summary (model under test)
# --------------------------------------------------------------------------
_SUMMARIZER_SYSTEM = textwrap.dedent("""\
    You are a meticulous legal summarizer. Produce a faithful summary of the document using only information it contains. Invent nothing, infer nothing, and add no outside legal knowledge, analysis, or recommendations.

    - Preserve every legally operative detail exactly as written: party names, dates, deadlines, monetary amounts, percentages, defined terms, governing law, jurisdiction, and any citations or section references. Keep these verbatim even when the summary is in another language; do not translate or reformat names, defined terms, citations, or numeric values.
    - Preserve the force of each provision. Keep obligations ("shall", "must"), permissions ("may"), prohibitions, conditions ("subject to", "provided that"), exceptions, and qualifiers intact; do not soften, strengthen, or generalize them.
    - Attribute statements to their source. Represent allegations, arguments, and positions as such (e.g. "Plaintiff alleges", "the Agreement provides", "the court held") rather than asserting them as established fact.
    - Cover the material content (key rights, obligations, conditions, deadlines, liabilities, and remedies) and prioritize it over boilerplate. Do not drop a term that changes the parties' rights or obligations.
    - Do not resolve ambiguity or infer unstated facts. If the document is unclear or silent on a point, do not fill the gap with a plausible guess.
    - Stay neutral: do not evaluate the document, predict outcomes, or give legal advice.""")


def summarize(target_words: int, language: str) -> tuple[str, str]:
    """Build (system, user) prompts for faithful legal-document summarization.

    The document text is supplied by the caller (appended after the user
    message). For benchmarking, call with identical target_words and language
    across every model so the prompts stay byte-identical and output
    differences reflect the model alone.
    """
    user = (
        f"Summarize the following document in approximately {target_words} words. "
        f"Language of the summary: {language}. Write exclusively in this language. "
        "Output only the summary, without any preamble."
    )
    return _SUMMARIZER_SYSTEM, user


def summarize_section(target_words: int | None, language: str) -> tuple[str, str]:
    """Summarize a single SECTION of a larger document.

    Same faithful-summarizer rules. With ``target_words=None`` the length is left
    adaptive (the summary is as short as the section allows without dropping
    material detail) -- mirroring a per-section production summarizer; pass a value
    only to force a per-section length (e.g. a fixed compression ratio).
    """
    length = (
        f"in approximately {target_words} words"
        if target_words
        else "as concisely as its content allows, without dropping any material detail"
    )
    user = (
        f"Summarize the following section of a larger document {length}. "
        f"Language of the summary: {language}. Write exclusively in this language. "
        "Output only the summary, without any preamble."
    )
    return _SUMMARIZER_SYSTEM, user


# --------------------------------------------------------------------------
# Answering with context (grounding) -- for ORIGINAL AND SUMMARY
# --------------------------------------------------------------------------
def answer_grounded(require_evidence: bool, language: str) -> tuple[str, str]:
    evidence_rule = (
        "In the 'evidence' field, provide a verbatim quote from the CONTEXT that "
        "supports your answer."
        if require_evidence
        else "The 'evidence' field may be left empty."
    )
    system = (
        "You are an extractive question-answering system. You answer the question "
        "EXCLUSIVELY based on the provided CONTEXT. You use NO prior knowledge and "
        "no information from outside the CONTEXT. "
        f"If the answer is not in the CONTEXT, set 'found' to false and 'answer' "
        f"to exactly '{NOT_IN_DOC}'. Otherwise set 'found' to true. "
        f"Language of the 'answer' field: {language} (the marker '{NOT_IN_DOC}' "
        "remains unchanged). "
        + evidence_rule
    )
    user = (
        "CONTEXT see above.\n\n"
        "QUESTION: {question}\n\n"
        "Answer exclusively as JSON according to the schema."
    )
    return system, user


# --------------------------------------------------------------------------
# Closed-book baseline (no context) -- contamination check
# --------------------------------------------------------------------------
def answer_closed_book(language: str) -> tuple[str, str]:
    system = (
        "You answer the question solely from your general knowledge. There is no "
        "context. If you do not know the answer for certain, set 'found' to false "
        f"and 'answer' to exactly '{NOT_KNOWN}'. Otherwise set 'found' to true. "
        f"Language of the 'answer' field: {language} (the marker '{NOT_KNOWN}' "
        "remains unchanged). "
        "The 'evidence' field stays empty."
    )
    user = "QUESTION: {question}\n\nAnswer exclusively as JSON according to the schema."
    return system, user


# --------------------------------------------------------------------------
# Judging (strong model as judge)
# --------------------------------------------------------------------------
def judge(scale: list[str]) -> tuple[str, str]:
    system = (
        "You are a strict, fair evaluator. You receive a question, a REFERENCE "
        "ANSWER (treated as true) and a CANDIDATE ANSWER. Judge how well the "
        "candidate answer matches the reference answer in content -- not word for "
        "word, but in meaning.\n"
        f"Possible verdicts: {', '.join(scale)}.\n"
        "- match: same core statement.\n"
        "- partial: partially correct or incomplete.\n"
        "- mismatch: wrong, contradictory or no answer "
        f"(e.g. '{NOT_IN_DOC}') although the reference has an answer."
    )
    user = (
        "QUESTION: {question}\n\n"
        "REFERENCE ANSWER: {reference}\n\n"
        "CANDIDATE ANSWER: {candidate}\n\n"
        "Return your verdict exclusively as JSON according to the schema."
    )
    return system, user
