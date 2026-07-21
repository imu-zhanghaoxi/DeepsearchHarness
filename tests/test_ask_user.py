"""Tests for ask_user interactive tool."""

import pytest

from src.core.tool import ToolUseContext
from src.tools.ask_user import AskUserTool


@pytest.fixture
def tool() -> AskUserTool:
    return AskUserTool()


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(cache_dir=tmp_path / "cache")


class TestAskUserValidateInput:
    def test_requires_question(self, tool: AskUserTool):
        result = tool.validate_input({"question": "", "options": [{"label": "A"}, {"label": "B"}]})
        assert result.valid is False

    def test_requires_at_least_two_options(self, tool: AskUserTool):
        result = tool.validate_input({"question": "Which?", "options": [{"label": "Only"}]})
        assert result.valid is False

    def test_accepts_valid_input(self, tool: AskUserTool):
        result = tool.validate_input(
            {
                "question": "Which region?",
                "options": [{"label": "US"}, {"label": "EU"}],
            }
        )
        assert result.valid is True


@pytest.mark.asyncio
async def test_call_returns_pending_question_metadata(tool: AskUserTool, context: ToolUseContext):
    result = await tool.call(
        {
            "question": "Which timeframe?",
            "options": [
                {"label": "Last week", "description": "Recent news"},
                {"label": "Last year", "description": "Broader context"},
            ],
        },
        context,
    )

    assert result.metadata["pending_question"]["question"] == "Which timeframe?"
    assert len(result.metadata["pending_question"]["options"]) == 2
    assert result.data == ""
