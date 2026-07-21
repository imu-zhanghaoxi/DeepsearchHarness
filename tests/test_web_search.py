"""Tests for SearXNG-backed web_search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.tool import ToolUseContext
from src.tools.web_search import WebSearchTool


@pytest.fixture
def tool(tmp_path) -> WebSearchTool:
    return WebSearchTool(
        searxng_url="http://searxng.test",
        default_results=5,
        max_results=10,
        max_result_size_chars=5000,
        http_timeout=5,
        engines="google,bing",
        language="en",
    )


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestWebSearchValidateInput:
    def test_rejects_short_query(self, tool: WebSearchTool):
        result = tool.validate_input({"query": "a"})
        assert result.valid is False

    def test_accepts_valid_query(self, tool: WebSearchTool):
        result = tool.validate_input({"query": "python asyncio"})
        assert result.valid is True


class TestWebSearchSearxng:
    @pytest.mark.asyncio
    async def test_search_searxng_builds_request(self, tool: WebSearchTool):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com",
                    "content": "snippet text",
                    "engine": "google",
                }
            ]
        }
        tool._client.get = AsyncMock(return_value=mock_response)

        results = await tool._search_searxng("python tutorials", 5)

        assert len(results) == 1
        assert results[0]["link"] == "https://example.com"
        assert results[0]["snippet"] == "snippet text"
        tool._client.get.assert_awaited_once()
        call_kwargs = tool._client.get.await_args.kwargs
        assert call_kwargs["params"]["q"] == "python tutorials"
        assert call_kwargs["params"]["format"] == "json"
        assert call_kwargs["params"]["engines"] == "google,bing"
        assert call_kwargs["params"]["language"] == "en"

    @pytest.mark.asyncio
    async def test_search_searxng_http_error_returns_empty(self, tool: WebSearchTool):
        request = httpx.Request("GET", "http://searxng.test/search")
        response = httpx.Response(403, request=request)
        tool._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("forbidden", request=request, response=response)
        )

        results = await tool._search_searxng("blocked query", 5)
        assert results == []


class TestWebSearchCall:
    @pytest.mark.asyncio
    async def test_call_formats_results_and_citations(
        self, tool: WebSearchTool, context: ToolUseContext
    ):
        with patch.object(
            tool,
            "_search_searxng",
            new=AsyncMock(
                return_value=[
                    {
                        "title": "Python Docs",
                        "link": "https://docs.python.org",
                        "snippet": "Official documentation",
                    }
                ]
            ),
        ):
            result = await tool.call({"query": "python docs"}, context)

        assert "Python Docs" in result.data
        assert "https://docs.python.org" in result.data
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://docs.python.org"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_call_no_results_message(self, tool: WebSearchTool, context: ToolUseContext):
        with patch.object(tool, "_search_searxng", new=AsyncMock(return_value=[])):
            result = await tool.call({"query": "nothing here"}, context)

        assert "No search results found" in result.data
        assert "searxng.test" in result.data
        assert result.citations == []

    @pytest.mark.asyncio
    async def test_call_respects_num_results_cap(
        self, tool: WebSearchTool, context: ToolUseContext
    ):
        with patch.object(tool, "_search_searxng", new=AsyncMock(return_value=[])) as mock_search:
            await tool.call({"query": "python", "num_results": 99}, context)

        mock_search.assert_awaited_once_with("python", 10)

    @pytest.mark.asyncio
    async def test_call_uses_rate_limiter(self, tool: WebSearchTool, context: ToolUseContext):
        limiter = AsyncMock()
        context.rate_limiter = limiter

        with patch.object(tool, "_search_searxng", new=AsyncMock(return_value=[])):
            await tool.call({"query": "python"}, context)

        limiter.acquire.assert_awaited_once_with("http://searxng.test")

    def test_to_api_schema(self, tool: WebSearchTool):
        schema = tool.to_api_schema()
        assert schema["function"]["name"] == "search_web"
        assert "query" in schema["function"]["parameters"]["properties"]
