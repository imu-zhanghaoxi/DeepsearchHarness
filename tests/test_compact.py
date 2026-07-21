"""Tests for context compaction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.core.compact import (
    CLEARED_TOOL_RESULT,
    _format_compact_summary,
    _microcompact,
    should_compact,
)
from src.core.types import Message


def test_should_compact_when_over_threshold():
    messages = [
        Message(role="user", content="x" * 400_000),
    ]

    assert should_compact(messages, threshold_tokens=10_000) is True


def test_should_not_compact_when_under_threshold():
    messages = [
        Message(role="user", content="short question"),
        Message(role="assistant", content="short answer"),
    ]

    assert should_compact(messages, threshold_tokens=80_000) is False


def test_microcompact_clears_old_large_tool_results():
    messages = [
        Message(role="user", content="query"),
        Message(role="assistant", content="searching"),
        Message(
            role="tool",
            content="x" * 1000,
            metadata={"tool_call_id": "1", "tool_name": "fetch_url"},
        ),
        Message(role="assistant", content="more"),
        Message(
            role="tool",
            content="y" * 1000,
            metadata={"tool_call_id": "2", "tool_name": "fetch_url"},
        ),
        Message(role="assistant", content="recent"),
        Message(role="user", content="follow up"),
    ]

    compacted = _microcompact(messages, keep_last_n=2)

    assert compacted[0].text_content == "query"
    assert compacted[2].text_content == CLEARED_TOOL_RESULT
    assert compacted[-1].text_content == "follow up"


def test_format_compact_summary_strips_analysis_block():
    raw = "<analysis>scratch</analysis><summary>Final summary body</summary>"

    assert _format_compact_summary(raw) == "Final summary body"


@pytest.mark.asyncio
async def test_compact_messages_uses_side_query_when_needed():
    from src.core.compact import compact_messages

    messages = [
        Message(role="user", content="What is Python?"),
        Message(role="assistant", content="Let me search."),
        Message(
            role="tool",
            content="z" * 2000,
            metadata={"tool_call_id": "1", "tool_name": "search_web"},
        ),
        Message(role="assistant", content="Python is a language."),
    ]

    with (
        patch("src.core.compact.should_compact", return_value=True),
        patch(
            "src.core.compact._microcompact",
            return_value=messages,
        ),
        patch(
            "src.llm.client.side_query",
            new_callable=AsyncMock,
            return_value="<summary>Summarized findings with https://example.com</summary>",
        ),
    ):
        compacted = await compact_messages(messages, threshold_tokens=100)

    assert len(compacted) >= 2
    assert compacted[0].role == "user"
    assert "Summarized findings" in compacted[1].text_content
