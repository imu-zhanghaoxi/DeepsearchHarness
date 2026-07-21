"""Tests for plan completeness stop hook."""

from __future__ import annotations

import pytest

from src.core.types import LoopState, Message, ResearchPlan, ResearchTask
from src.hooks.plan_completeness_hook import PlanCompletenessHook


@pytest.fixture
def hook() -> PlanCompletenessHook:
    return PlanCompletenessHook()


@pytest.mark.asyncio
async def test_passes_when_no_plan(hook: PlanCompletenessHook):
    state = LoopState(messages=[Message(role="user", content="Hello")], turn_count=1, citations=[])

    result = await hook.evaluate(state)

    assert result.passed is True


@pytest.mark.asyncio
async def test_passes_when_plan_complete(hook: PlanCompletenessHook):
    plan = ResearchPlan(
        tasks=[
            ResearchTask(id="1", title="A", status="completed"),
            ResearchTask(id="2", title="B", status="completed"),
        ]
    )
    state = LoopState(
        messages=[Message(role="user", content="Compare A and B")],
        turn_count=2,
        citations=[],
        research_plan=plan,
    )

    result = await hook.evaluate(state)

    assert result.passed is True


@pytest.mark.asyncio
async def test_fails_when_tasks_remain(hook: PlanCompletenessHook):
    plan = ResearchPlan(
        tasks=[
            ResearchTask(id="1", title="A", status="completed"),
            ResearchTask(id="2", title="B", status="pending"),
        ]
    )
    state = LoopState(
        messages=[Message(role="user", content="Compare A and B")],
        turn_count=2,
        citations=[],
        research_plan=plan,
    )

    result = await hook.evaluate(state)

    assert result.passed is False
    assert "incomplete sub-task" in result.feedback.lower()
    assert "[2] B" in result.feedback
