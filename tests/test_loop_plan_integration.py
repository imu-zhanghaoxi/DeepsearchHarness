"""Loop integration tests for research plan events and plan completeness hook."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from src.core.loop import QueryParams, query_loop
from src.core.tool import ToolRegistry
from src.core.types import EventType, StreamEvent
from src.hooks.engine import HookEngine
from src.hooks.plan_completeness_hook import PlanCompletenessHook
from src.llm.client import LLMClient, ModelConfig
from src.tools.research_plan import ResearchPlanTool


class FakeLLMClient(LLMClient):
    def __init__(self, script: list[list[StreamEvent]]):
        super().__init__(ModelConfig())
        self._script = list(script)
        self._call_index = 0

    async def stream(
        self, messages, system_prompt, tools=None, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        del messages, system_prompt, tools, kwargs
        events = self._script[self._call_index]
        self._call_index += 1
        for event in events:
            yield event


@pytest.mark.asyncio
async def test_loop_emits_plan_update_after_research_plan_tool():
    registry = ToolRegistry()
    registry.register(ResearchPlanTool())

    llm = FakeLLMClient(
        script=[
            [
                StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": "call_1",
                        "tool_name": "research_plan",
                        "tool_input": {
                            "action": "create",
                            "tasks": [{"title": "Aspect A"}, {"title": "Aspect B"}],
                        },
                    },
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": "I created a plan and will continue later."},
                ),
            ],
        ]
    )

    params = QueryParams(
        query="Compare A and B",
        system_prompt="sys",
        tool_registry=registry,
        llm_client=llm,
        max_turns=5,
        hook_engine=None,
    )

    events = [event async for event in query_loop(params)]

    assert EventType.PLAN_UPDATE in [event.type for event in events]
    plan_event = next(event for event in events if event.type == EventType.PLAN_UPDATE)
    assert plan_event.data["total_count"] == 2


@pytest.mark.asyncio
async def test_plan_completeness_hook_blocks_finalize_until_complete():
    registry = ToolRegistry()
    registry.register(ResearchPlanTool())

    engine = HookEngine()
    engine.register_stop_hook(PlanCompletenessHook())

    llm = FakeLLMClient(
        script=[
            [
                StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": "call_1",
                        "tool_name": "research_plan",
                        "tool_input": {
                            "action": "create",
                            "tasks": [{"title": "Aspect A"}],
                        },
                    },
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": "Here is my premature final answer without finishing the plan."},
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": "call_2",
                        "tool_name": "research_plan",
                        "tool_input": {
                            "action": "update",
                            "task_id": "1",
                            "status": "completed",
                            "findings": "Done researching A.",
                        },
                    },
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": "Final answer after completing the research plan task."},
                ),
            ],
        ]
    )

    params = QueryParams(
        query="Explain aspect A",
        system_prompt="sys",
        tool_registry=registry,
        llm_client=llm,
        hook_engine=engine,
        max_turns=8,
    )

    events = [event async for event in query_loop(params)]
    status_messages = [
        event.data.get("message", "") for event in events if event.type == EventType.STATUS
    ]

    assert any("Quality check" in message for message in status_messages)
    done = next(event for event in events if event.type == EventType.DONE)
    assert "after completing" in done.data["final_answer"]
