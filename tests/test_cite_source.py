"""Tests for cite_source tool."""

from __future__ import annotations

import pytest

from src.core.tool import ToolUseContext
from src.core.types import SourceType
from src.tools.cite_source import CiteSourceTool


@pytest.fixture
def tool() -> CiteSourceTool:
    return CiteSourceTool()


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestCiteSourceValidateInput:
    def test_rejects_missing_url(self, tool: CiteSourceTool):
        result = tool.validate_input({"title": "T", "snippet": "S"})
        assert result.valid is False

    def test_rejects_missing_title(self, tool: CiteSourceTool):
        result = tool.validate_input({"url": "https://example.com", "snippet": "S"})
        assert result.valid is False

    def test_rejects_missing_snippet(self, tool: CiteSourceTool):
        result = tool.validate_input({"url": "https://example.com", "title": "T"})
        assert result.valid is False

    def test_accepts_valid_args(self, tool: CiteSourceTool):
        result = tool.validate_input(
            {
                "url": "https://example.com",
                "title": "Example",
                "snippet": "A relevant quote.",
            }
        )
        assert result.valid is True


class TestCiteSourceCall:
    @pytest.mark.asyncio
    async def test_registers_citation_with_cited_true(
        self, tool: CiteSourceTool, context: ToolUseContext
    ):
        result = await tool.call(
            {
                "url": "https://example.com/page",
                "title": "Example Page",
                "snippet": "Key fact from the page.",
                "source_type": "web",
                "relevance_note": "Supports the main claim.",
            },
            context,
        )

        assert result.is_error is False
        assert len(result.citations) == 1
        citation = result.citations[0]
        assert citation.url == "https://example.com/page"
        assert citation.title == "Example Page"
        assert citation.snippet == "Key fact from the page."
        assert citation.source_type == SourceType.WEB
        assert citation.cited is True
        assert "Example Page" in result.data
        assert "Supports the main claim" in result.data
