"""Tests for deep_read tool."""

from pathlib import Path

import pytest

from src.core.tool import ToolUseContext
from src.tools.deep_read import DeepReadTool


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    return cache


@pytest.fixture
def tool() -> DeepReadTool:
    return DeepReadTool(max_result_size_chars=1000)


@pytest.fixture
def context(cache_dir: Path) -> ToolUseContext:
    return ToolUseContext(session_id="test", cache_dir=cache_dir)


@pytest.mark.asyncio
async def test_reads_line_range(tool: DeepReadTool, context: ToolUseContext, cache_dir: Path):
    cached = cache_dir / "page.md"
    cached.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    result = await tool.call(
        {"cached_path": str(cached), "start_line": 2, "end_line": 3},
        context,
    )

    assert not result.is_error
    assert "line2" in result.data
    assert "line3" in result.data
    assert "line4" not in result.data


@pytest.mark.asyncio
async def test_extracts_heading_section(
    tool: DeepReadTool, context: ToolUseContext, cache_dir: Path
):
    cached = cache_dir / "doc.md"
    cached.write_text(
        "# Introduction\nintro text\n\n## Methodology\nmethod details\n\n## Results\nresult data\n",
        encoding="utf-8",
    )

    result = await tool.call(
        {"cached_path": str(cached), "section_query": "methodology"},
        context,
    )

    assert not result.is_error
    assert "method details" in result.data
    assert "result data" not in result.data


@pytest.mark.asyncio
async def test_blocks_path_traversal(tool: DeepReadTool, context: ToolUseContext, tmp_path: Path):
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    result = await tool.call({"cached_path": str(outside)}, context)

    assert result.is_error
    assert "Access denied" in result.data


@pytest.mark.asyncio
async def test_missing_file_returns_error(
    tool: DeepReadTool, context: ToolUseContext, cache_dir: Path
):
    missing = cache_dir / "missing.md"
    result = await tool.call({"cached_path": str(missing)}, context)

    assert result.is_error
    assert "Failed to read" in result.data
