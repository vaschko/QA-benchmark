"""Document language detection.

We detect the language once from the original document and then name it
*explicitly* to the models in the prompt (e.g. "English"). A mere instruction
"in the language of the document" is often ignored by smaller models -- a
concrete language name is followed reliably.
"""

from __future__ import annotations

# Language code (ISO 639-1, as returned by langdetect) -> English name.
_NAMES: dict[str, str] = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "el": "Greek",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ar": "Arabic",
}

_FALLBACK = "the original language of the document"


def detect_language(text: str) -> str:
    """Returns a language name ready to insert into the prompt (e.g. 'English').
    Falls back to a neutral phrase when detection is uncertain."""
    sample = text.strip()[:2000]
    if not sample:
        return _FALLBACK
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0  # deterministic detection
        code = detect(sample)
    except Exception:
        return _FALLBACK
    return _NAMES.get(code, _FALLBACK)
