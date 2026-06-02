"""QA-Benchmark: measures how well summaries preserve information.

Core idea: questions are generated from the original and answered both with the
original (= reference) and with the summary (= candidate). A strong judge model
compares the two; the retention score indicates how many questions the summary
lets you answer just as well as the original.
"""

__version__ = "0.1.0"
