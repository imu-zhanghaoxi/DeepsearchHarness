"""Tests for SearXNG-backed news_search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from src.core.tool import ToolUseContext
from src.core.types import SourceType
from src.tools.news_search import NewsSearchTool, _time_range_for_days


@pytest.fixture
def tool(tmp_path) -> NewsSearchTool:
    return NewsSearchTool(
        searxng_url="http://searxng.test",
        default_results=5,
        max_results=10,
        default_days_back=7,
        max_days_back=30,
        engines="bing news",
        language="en",
    )


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestNewsSearchTimeRange:
    def test_time_range_mapping(self):
        assert _time_range_for_days(1) == "day"
        assert _time_range_for_days(7) == "week"
        assert _time_range_for_days(14) == "month"
        assert _time_range_for_days(60) == "year"


class TestNewsSearchValidateInput:
    def test_rejects_short_query(self, tool: NewsSearchTool):
        assert tool.validate_input({"query": "a"}).valid is False

    def test_accepts_valid_query(self, tool: NewsSearchTool):
        assert tool.validate_input({"query": "AI regulation"}).valid is True


class TestNewsSearchSearxng:
    @pytest.mark.asyncio
    async def test_search_uses_news_category(self, tool: NewsSearchTool):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Breaking: AI News",
                    "url": "https://news.example.com/1",
                    "content": "Summary text",
                    "engine": "bing news",
                    "publishedDate": "2024-06-01T10:00:00",
                }
            ]
        }
        tool._client.get = AsyncMock(return_value=mock_response)

        articles = await tool._search_searxng_news("AI news", 5, 7)

        assert len(articles) == 1
        assert articles[0]["url"] == "https://news.example.com/1"
        call_kwargs = tool._client.get.await_args.kwargs
        assert call_kwargs["params"]["categories"] == "news"
        assert call_kwargs["params"]["time_range"] == "week"
        assert call_kwargs["params"]["engines"] == "bing news"

    @pytest.mark.asyncio
    async def test_search_http_error_returns_empty(self, tool: NewsSearchTool):
        request = httpx.Request("GET", "http://searxng.test/search")
        response = httpx.Response(503, request=request)
        tool._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("unavailable", request=request, response=response)
        )

        articles = await tool._search_searxng_news("query", 5, 7)
        assert articles == []


class TestNewsSearchCall:
    @pytest.mark.asyncio
    async def test_call_formats_results(self, tool: NewsSearchTool, context: ToolUseContext):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Market Update",
                    "url": "https://news.example.com/market",
                    "content": "Markets rose today.",
                    "engine": "google news",
                }
            ]
        }
        tool._client.get = AsyncMock(return_value=mock_response)

        result = await tool.call({"query": "stock market"}, context)

        assert not result.is_error
        assert "Market Update" in result.data
        assert len(result.citations) == 1
        assert result.citations[0].source_type == SourceType.NEWS

    @pytest.mark.asyncio
    async def test_call_no_results_message(self, tool: NewsSearchTool, context: ToolUseContext):
        tool._client.get = AsyncMock(
            return_value=MagicMock(
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"results": []}),
            )
        )

        result = await tool.call({"query": "obscure topic"}, context)

        assert not result.is_error
        assert "No recent news articles found" in result.data
