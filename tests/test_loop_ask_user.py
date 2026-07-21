"""Loop integration test for ask_user USER_QUESTION flow."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from src.core.loop import QueryParams, query_loop
from src.core.tool import ToolRegistry
from src.core.types import EventType, StreamEvent
from src.llm.client import LLMClient, ModelConfig
from src.tools.ask_user import AskUserTool


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
            yield StreamEvent(type=EventType.TEXT_DELTA, data={"text": "Final answer."})
            return
        events = self._script[self._call_index]
        self._call_index += 1
        for event in events:
            yield event


@pytest.mark.asyncio
async def test_loop_ask_user_receives_answer_via_asend():
    registry = ToolRegistry()
    registry.register(AskUserTool())

    llm = FakeLLMClient(
        script=[
            [
                StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": "call_ask",
                        "tool_name": "ask_user",
                        "tool_input": {
                            "question": "Which focus?",
                            "options": [{"label": "Technical"}, {"label": "Business"}],
                        },
                    },
                ),
            ],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": "Here is the technical answer."},
                ),
            ],
        ]
    )

    params = QueryParams(
        query="Tell me about AI",
        system_prompt="You are a researcher.",
        tool_registry=registry,
        llm_client=llm,
        max_turns=5,
    )

    gen = query_loop(params)
    events: list[StreamEvent] = []
    sent_value: str | None = None

    while True:
        try:
            event = await gen.asend(sent_value)
        except StopAsyncIteration:
            break
        sent_value = None
        events.append(event)
        if event.type == EventType.USER_QUESTION:
            assert event.data["question"] == "Which focus?"
            sent_value = "Technical"

    assert any(e.type == EventType.USER_QUESTION for e in events)
    assert any(e.type == EventType.TOOL_RESULT for e in events)
    tool_result = next(e for e in events if e.type == EventType.TOOL_RESULT)
    assert "Technical" in tool_result.data["result"]
    done = next(e for e in events if e.type == EventType.DONE)
    assert "technical answer" in done.data["final_answer"]
