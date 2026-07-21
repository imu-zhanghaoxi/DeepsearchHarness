"""Tests for content_extractor utility."""

from unittest.mock import AsyncMock, patch

import pytest

from src.utils.content_extractor import extract_content


class TestExtractContent:
    @pytest.mark.asyncio
    async def test_returns_none_without_research_query(self):
        result = await extract_content(
            raw_content="long content",
            research_query="",
            source_url="https://example.com",
            source_title="Example",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_side_query_and_returns_summary(self):
        with patch(
            "src.utils.content_extractor.side_query",
            new=AsyncMock(return_value="## Example\n**Source**: https://example.com\n\nFacts here."),
        ) as mock_side_query:
            result = await extract_content(
                raw_content="raw " * 100,
                research_query="What happened?",
                source_url="https://example.com",
                source_title="Example",
            )

        assert result is not None
        assert "Facts here" in result
        mock_side_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_side_query_too_short(self):
        with patch(
            "src.utils.content_extractor.side_query",
            new=AsyncMock(return_value="short"),
        ):
            result = await extract_content(
                raw_content="raw " * 100,
                research_query="What happened?",
                source_url="https://example.com",
                source_title="Example",
            )

        assert result is None
