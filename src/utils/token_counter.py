"""
Token estimation utilities.

Uses tiktoken for fast, accurate token counting. Falls back to
a word-based heuristic if tiktoken is unavailable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            import tiktoken

            _encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktoken not available, using heuristic: {e}")
            _encoder = "heuristic"
    return _encoder


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    if not text:
        return 0

    encoder = _get_encoder()
    if encoder == "heuristic":
        return len(text) // 4

    try:
        return len(encoder.encode(text))
    except Exception:
        return len(text) // 4
