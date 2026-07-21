"""Tests for research_plan tool."""

from __future__ import annotations

import pytest

from src.core.tool import ToolUseContext
from src.core.types import LoopState
from src.tools.research_plan import ResearchPlanTool


@pytest.fixture
def tool() -> ResearchPlanTool:
    return ResearchPlanTool()


@pytest.fixture
def context(tmp_path) -> ToolUseContext:
    return ToolUseContext(
        cache_dir=tmp_path / "cache",
        extra={"loop_state": LoopState(messages=[], turn_count=1, citations=[])},
    )


class TestResearchPlanValidateInput:
    def test_rejects_invalid_action(self, tool: ResearchPlanTool):
        result = tool.validate_input({"action": "delete"})
        assert result.valid is False

    def test_create_requires_tasks(self, tool: ResearchPlanTool):
        result = tool.validate_input({"action": "create"})
        assert result.valid is False

    def test_update_requires_task_id_and_status(self, tool: ResearchPlanTool):
        result = tool.validate_input({"action": "update", "task_id": "1"})
        assert result.valid is False


class TestResearchPlanCall:
    @pytest.mark.asyncio
    async def test_create_plan(self, tool: ResearchPlanTool, context: ToolUseContext):
        result = await tool.call(
            {
                "action": "create",
                "tasks": [
                    {"title": "Topic A"},
                    {"title": "Topic B", "details": "Focus on recent sources"},
                ],
            },
            context,
        )

        loop_state = context.extra["loop_state"]
        assert result.is_error is False
        assert loop_state.research_plan is not None
        assert len(loop_state.research_plan.tasks) == 2
        assert loop_state.research_plan.tasks[0].id == "1"
        assert "Research plan created" in result.data

    @pytest.mark.asyncio
    async def test_update_and_check_plan(self, tool: ResearchPlanTool, context: ToolUseContext):
        await tool.call(
            {
                "action": "create",
                "tasks": [{"title": "Topic A"}, {"title": "Topic B"}],
            },
            context,
        )

        update_result = await tool.call(
            {
                "action": "update",
                "task_id": "1",
                "status": "completed",
                "findings": "Found key facts about topic A.",
            },
            context,
        )
        assert update_result.is_error is False
        assert context.extra["loop_state"].research_plan.completed_count == 1

        check_result = await tool.call({"action": "check"}, context)
        assert check_result.is_error is False
        assert "1 sub-task(s) remaining" in check_result.data

    @pytest.mark.asyncio
    async def test_complete_plan_reports_ready(self, tool: ResearchPlanTool, context: ToolUseContext):
        await tool.call(
            {"action": "create", "tasks": [{"title": "Only task"}]},
            context,
        )
        result = await tool.call(
            {
                "action": "update",
                "task_id": "1",
                "status": "completed",
                "findings": "Done.",
            },
            context,
        )

        assert context.extra["loop_state"].research_plan.is_complete is True
        assert "All sub-tasks completed" in result.data
