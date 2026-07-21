"""P0 agent loop tests with fake LLM and fake tools."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from src.core.loop import QueryParams, query_loop
from src.core.tool import Tool, ToolRegistry, ToolUseContext
from src.core.types import Citation, EventType, SourceType, StreamEvent, ToolResult
from src.llm.client import LLMClient, ModelConfig


class FakeSearchTool(Tool):
    name = "search_web"
    description = "Fake search"
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }
    is_concurrency_safe = True

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = args["query"]
        return ToolResult(
            data=f"## Search Results for: {query}\n\n### 1. Example\n**URL**: https://example.com\n",
            citations=[
                Citation(
                    url="https://example.com",
                    title="Example",
                    snippet="Example snippet",
                    source_type=SourceType.WEB,
                )
            ],
        )


class FakeLLMClient(LLMClient):
    def __init__(self, script: list[list[StreamEvent]]):
        super().__init__(ModelConfig())
        self._script = list(script)
        self._call_index = 0

    async def stream(
        self, messages, system_prompt, tools=None, **kwargs
    ) -> AsyncGenerator[StreamEvent, None]:
        del messages, system_prompt, tools, kwargs
        if self._call_index >= len(self._script):
            yield StreamEvent(type=EventType.TEXT_DELTA, data={"text": "Default final answer."})
            return

        events = self._script[self._call_index]
        self._call_index += 1
        for event in events:
            yield event


@pytest.mark.asyncio
async def test_loop_search_then_answer():
    registry = ToolRegistry()
    registry.register(FakeSearchTool())

    llm = FakeLLMClient(
        script=[
            [
                StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": "call_1",
                        "tool_name": "search_web",
                        "tool_input": {"query": "python asyncio"},
                    },
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={
                        "text": (
                            "Python asyncio helps write concurrent code. "
                            "See [Example](https://example.com)."
                        ),
                    },
                ),
            ],
        ]
    )

    params = QueryParams(
        query="What is Python asyncio?",
        system_prompt="You are a researcher.",
        tool_registry=registry,
        llm_client=llm,
        max_turns=5,
    )

    events = [event async for event in query_loop(params)]
    event_types = [event.type for event in events]

    assert EventType.TOOL_USE in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.CITATION in event_types
    assert EventType.DONE in event_types

    done = next(event for event in events if event.type == EventType.DONE)
    assert "asyncio" in done.data["final_answer"]
    assert len(done.data["citations"]) == 1
    assert done.data["turn_count"] == 2


@pytest.mark.asyncio
async def test_loop_immediate_answer_without_tools():
    registry = ToolRegistry()
    registry.register(FakeSearchTool())

    llm = FakeLLMClient(
        script=[
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": "Here is the direct answer."},
                ),
            ],
        ]
    )

    params = QueryParams(
        query="Simple question",
        system_prompt="You are helpful.",
        tool_registry=registry,
        llm_client=llm,
    )

    events = [event async for event in query_loop(params)]
    done = next(event for event in events if event.type == EventType.DONE)

    assert done.data["final_answer"] == "Here is the direct answer."
    assert done.data["turn_count"] == 1
