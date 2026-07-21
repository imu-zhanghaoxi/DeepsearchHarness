"""Tests for academic_search tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.tool import ToolUseContext
from src.core.types import SourceType
from src.tools.academic_search import AcademicSearchTool

SEMANTIC_SCHOLAR_RESPONSE = {
    "data": [
        {
            "title": "Attention Is All You Need",
            "authors": [{"name": "Vaswani"}],
            "abstract": "We propose the Transformer architecture.",
            "year": 2017,
            "citationCount": 100000,
            "url": "https://example.com/paper1",
            "venue": "NeurIPS",
        }
    ]
}

ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>  BERT: Pre-training </title>
    <id>https://arxiv.org/abs/1810.04805</id>
    <published>2018-10-11T00:00:00Z</published>
    <summary>Language model pre-training.</summary>
    <author><name>Devlin</name></author>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
</feed>"""


@pytest.fixture
def tool(tmp_path) -> AcademicSearchTool:
    return AcademicSearchTool(default_results=3, max_results=5, http_timeout=5)


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestAcademicSearchValidateInput:
    def test_rejects_short_query(self, tool: AcademicSearchTool):
        assert tool.validate_input({"query": "a"}).valid is False

    def test_accepts_valid_query(self, tool: AcademicSearchTool):
        assert tool.validate_input({"query": "transformer"}).valid is True


class TestAcademicSearchSemanticScholar:
    @pytest.mark.asyncio
    async def test_search_semantic_scholar(self, tool: AcademicSearchTool):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = SEMANTIC_SCHOLAR_RESPONSE
        tool._client.get = AsyncMock(return_value=mock_response)

        papers = await tool._search_semantic_scholar("transformer", 3)

        assert len(papers) == 1
        assert papers[0]["_source"] == "Semantic Scholar"
        assert papers[0]["title"] == "Attention Is All You Need"


class TestAcademicSearchArxiv:
    @pytest.mark.asyncio
    async def test_search_arxiv_parses_xml(self, tool: AcademicSearchTool):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.text = ARXIV_XML
        tool._client.get = AsyncMock(return_value=mock_response)

        papers = await tool._search_arxiv("bert", 3)

        assert len(papers) == 1
        assert papers[0]["_source"] == "arXiv"
        assert "BERT" in papers[0]["title"]
        assert papers[0]["year"] == 2018


class TestAcademicSearchCall:
    @pytest.mark.asyncio
    async def test_call_single_source(self, tool: AcademicSearchTool, context: ToolUseContext):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = SEMANTIC_SCHOLAR_RESPONSE
        tool._client.get = AsyncMock(return_value=mock_response)

        result = await tool.call(
            {"query": "transformer", "source": "semantic_scholar"},
            context,
        )

        assert not result.is_error
        assert "Attention Is All You Need" in result.data
        assert len(result.citations) == 1
        assert result.citations[0].source_type == SourceType.ACADEMIC

    @pytest.mark.asyncio
    async def test_deduplicate_prefers_abstract(self, tool: AcademicSearchTool):
        papers = [
            {"title": "Same Paper", "abstract": "", "citationCount": 10},
            {"title": "same paper", "abstract": "Has abstract", "citationCount": None},
        ]
        deduped = tool._deduplicate(papers)
        assert len(deduped) == 1
        assert deduped[0]["abstract"] == "Has abstract"
