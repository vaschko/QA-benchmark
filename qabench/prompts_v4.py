"""Central, versioned prompt templates (English).

All prompts in one place so the methodology is easy to revise and audit. The
grounding prompts are intentionally strict in order to suppress the model's
parametric prior knowledge.
"""

from __future__ import annotations

PROMPTS_VERSION = "4"

NOT_IN_DOC = "NOT_IN_DOCUMENT"
NOT_KNOWN = "NOT_KNOWN"


# --------------------------------------------------------------------------
# Question generation (strong model)
# --------------------------------------------------------------------------
def question_gen(count: int, types: list[str], language: str) -> tuple[str, str]:
    system = (
        "You are an exam designer. From a given document you create questions "
        "that can be answered exclusively from the content of that document. "
        "Good questions are specific and target concrete facts, numbers, names, "
        "dates, definitions or relationships in the document. Avoid questions "
        "that anyone could answer from general knowledge without the document, "
        "and questions whose answer is not in the document."
    )
    user = (
        f"Create exactly {count} questions about the following document. "
        f"Spread the questions across these types where possible: {', '.join(types)}. "
        "Cover different parts of the document. "
        f"IMPORTANT: Language of the questions: {language}. Write ALL questions "
        "exclusively in this language, regardless of the language of this instruction. "
        "Return the result exclusively as JSON according to the schema."
    )
    return system, user


# --------------------------------------------------------------------------
# Summary (model under test)
# --------------------------------------------------------------------------
def summarize(target_words: int, language: str) -> tuple[str, str]:
    system = (
        "You are a precise summarizer. You render the document's content "
        "faithfully and invent nothing. Important facts, numbers, names and "
        "dates are preserved."
    )
    user = (
        f"Summarize the following document in approximately {target_words} words. "
        f"Language of the summary: {language}. Write exclusively in this language. "
        "Output only the summary, without any preamble."
    )
    return system, user


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
