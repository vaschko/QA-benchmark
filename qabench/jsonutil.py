"""Robust parsing of JSON from model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(raw: str) -> Any:
    """Tries to extract a JSON object from a (possibly noisy) model response.

    With structured outputs (Ollama `format`, Anthropic tool-use), `raw` is
    already clean JSON. As a fallback, the first {...} or [...] block in the
    text is searched for.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON block via bracket matching.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == opener:
                depth += 1
            elif raw[i] == closer:
                depth -= 1
                if depth == 0:
                    candidate = raw[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from the model response:\n{raw[:500]}")
