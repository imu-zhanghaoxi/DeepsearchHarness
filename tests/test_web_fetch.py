"""Tests for web_fetch tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.tool import ToolUseContext
from src.tools.web_fetch import WebFetchTool, _html_to_markdown, _strip_tags


@pytest.fixture
def tool() -> WebFetchTool:
    return WebFetchTool(
        max_result_size_chars=500,
        http_timeout=5,
        jina_timeout=5,
        extraction_threshold=200,
    )


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestHtmlToMarkdown:
    def test_strip_tags_fallback(self):
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        text = _strip_tags(html)
        assert "Hello" in text
        assert "world" in text
        assert "<" not in text

    def test_html_to_markdown_uses_trafilatura_when_available(self):
        html = (
            "<html><head><title>x</title></head><body>"
            "<article><p>" + ("Important content. " * 20) + "</p></article>"
            "</body></html>"
        )
        result = _html_to_markdown(html, url="https://example.com")
        assert "Important content" in result


class TestWebFetchValidateInput:
    def test_blocks_private_url(self, tool: WebFetchTool):
        result = tool.validate_input({"url": "http://127.0.0.1/"})
        assert result.valid is False
        assert "SSRF" in result.message

    def test_requires_http_scheme(self, tool: WebFetchTool):
        result = tool.validate_input({"url": "ftp://example.com"})
        assert result.valid is False


class TestWebFetchCall:
    @pytest.mark.asyncio
    async def test_blocks_search_engine_urls(self, tool: WebFetchTool, context: ToolUseContext):
        result = await tool.call(
            {"url": "https://www.google.com/search?q=python"},
            context,
        )
        assert "search_web" in result.data
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_direct_fetch_converts_html(self, tool: WebFetchTool, context: ToolUseContext):
        html = (
            "<html><head><title>Example Page</title></head><body>"
            "<article><p>" + ("Direct fetch content. " * 20) + "</p></article>"
            "</body></html>"
        )
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.text = html
        tool._client.get = AsyncMock(return_value=mock_response)

        with patch.object(tool, "_fetch_via_jina", new=AsyncMock(return_value=None)):
            result = await tool.call({"url": "https://example.com/page"}, context)

        assert result.is_error is False
        assert "Example Page" in result.data or "Direct fetch content" in result.data
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/page"

    @pytest.mark.asyncio
    async def test_jina_success_short_circuits_direct_fetch(
        self, tool: WebFetchTool, context: ToolUseContext
    ):
        jina_result = MagicMock()
        jina_result.data = "## Jina Title\n**Source**: https://example.com\n\nBody"
        jina_result.citations = []
        jina_result.truncated = False
        jina_result.cached_path = None
        jina_result.is_error = False

        with patch.object(tool, "_fetch_via_jina", new=AsyncMock(return_value=jina_result)):
            tool._client.get = AsyncMock()
            result = await tool.call({"url": "https://example.com"}, context)

        assert result is jina_result
        tool._client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_direct_fetch_http_error(self, tool: WebFetchTool, context: ToolUseContext):
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(404, request=request)
        tool._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("not found", request=request, response=response)
        )

        with patch.object(tool, "_fetch_via_jina", new=AsyncMock(return_value=None)):
            result = await tool.call({"url": "https://example.com/missing"}, context)

        assert result.is_error is True
        assert "404" in result.data


class TestWebFetchLongContent:
    @pytest.mark.asyncio
    async def test_llm_extraction_when_content_exceeds_threshold(
        self, tool: WebFetchTool, context: ToolUseContext
    ):
        context.extra["research_query"] = "What is Python?"
        long_body = "x" * 500
        full_content = f"## Long Page\n**Source**: https://example.com\n\n{long_body}"

        with patch(
            "src.utils.content_extractor.extract_content",
            new=AsyncMock(return_value="## Summary\n**Source**: https://example.com\n\nKey facts."),
        ):
            result = await tool._maybe_extract_or_truncate(
                full_content,
                "https://example.com",
                "Long Page",
                context,
            )

        assert result is not None
        assert result.truncated is True
        assert "Key facts" in result.data
        assert result.cached_path is not None
        assert "deep_read" in result.data

    @pytest.mark.asyncio
    async def test_falls_back_to_truncation_when_extraction_fails(
        self, tool: WebFetchTool, context: ToolUseContext
    ):
        context.extra["research_query"] = "What is Python?"
        long_body = "y" * 500
        full_content = f"## Long Page\n**Source**: https://example.com\n\n{long_body}"

        with patch(
            "src.utils.content_extractor.extract_content",
            new=AsyncMock(return_value=None),
        ):
            result = await tool._maybe_extract_or_truncate(
                full_content,
                "https://example.com",
                "Long Page",
                context,
            )

        assert result is not None
        assert result.truncated is True
        assert "Content truncated" in result.data
        assert len(result.data) <= tool.max_result_size_chars + 200

    @pytest.mark.asyncio
    async def test_small_content_skips_extraction(
        self, tool: WebFetchTool, context: ToolUseContext
    ):
        result = await tool._maybe_extract_or_truncate(
            "## Short\n**Source**: https://example.com\n\nhi",
            "https://example.com",
            "Short",
            context,
        )
        assert result is None
