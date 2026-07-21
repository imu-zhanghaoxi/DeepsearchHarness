"""Tests for final citation filtering in the query loop."""

from __future__ import annotations

from src.core.loop import _final_citations_for_answer
from src.core.types import Citation


def test_final_citations_prefers_urls_in_answer():
    citations = [
        Citation(url="https://a.com", title="A", snippet="a", cited=False),
        Citation(url="https://b.com", title="B", snippet="b", cited=False),
    ]
    answer = "See [A](https://a.com) for details."

    final = _final_citations_for_answer(citations, answer)

    assert len(final) == 1
    assert final[0].url == "https://a.com"


def test_final_citations_falls_back_to_explicit_cite_source():
    citations = [
        Citation(url="https://a.com", title="A", snippet="a", cited=False),
        Citation(url="https://b.com", title="B", snippet="b", cited=True),
    ]
    answer = "Python asyncio helps write concurrent code."

    final = _final_citations_for_answer(citations, answer)

    assert len(final) == 1
    assert final[0].url == "https://b.com"
    assert final[0].cited is True
