import string
from typing import Optional, Set


_DEFAULT_EXCLUDED_POS: Set[str] = set([
    "ADP", "CCONJ", "SCONJ", "DET", "AUX", "PUNCT", "PART",
])

_MANUAL_STOPWORDS: Set[str] = set([
    "the", "and", "a", "an", "to", "of", "in", "on", "for", "with",
    "as", "at", "by", "from", "or", "but", "if", "is", "are", "be",
    "was", "were", "this", "that", "these", "those", "it", "i", "no",
    "her", "his", "he", "she", "they", "we", "you", "me", "him", "them",
    "us", "my", "your", "their", "our", "its"," "
])

_SPACY_NLP = None  # lazy global cache


def _ensure_spacy() -> None:
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return
    try:
        import spacy  # type: ignore
        _SPACY_NLP = spacy.load("en_core_web_sm")
    except Exception:
        _SPACY_NLP = None


def should_filter_by_pos(
    text: str,
    excluded_pos: Optional[Set[str]] = None,
    manual_stopwords: Optional[Set[str]] = None,
) -> bool:
    """
    Determine whether a token text should be filtered as a function word.

    Filtering rules:
    1) Empty/whitespace only → keep
    2) Pure punctuation → filter
    3) Manual stopwords (lowercased) → filter
    4) spaCy STOP_WORDS → filter (if available)
    5) POS tag in excluded_pos → filter (if spaCy model available)
    """
    stripped = (text or "").strip()
    if not stripped:
        return False

    # pure punctuation
    if all(ch in string.punctuation for ch in stripped):
        return True

    excluded = excluded_pos if excluded_pos is not None else _DEFAULT_EXCLUDED_POS
    stopwords = manual_stopwords if manual_stopwords is not None else _MANUAL_STOPWORDS

    norm = stripped.lower()
    if norm in stopwords:
        return True

    # spaCy STOP_WORDS without loading model
    try:
        from spacy.lang.en import STOP_WORDS  # type: ignore
        if norm in STOP_WORDS:
            return True
    except Exception:
        pass

    # POS filtering (if model available)
    _ensure_spacy()
    if _SPACY_NLP is None:
        return False
    try:
        doc = _SPACY_NLP(stripped)
        if len(doc) != 1:
            return False
        return doc[0].pos_ in excluded
    except Exception:
        return False















