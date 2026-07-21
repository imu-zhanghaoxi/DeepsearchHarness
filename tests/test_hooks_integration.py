"""Tests for hook engine integration with the query loop."""

from __future__ import annotations

from typing import AsyncGenerator

import pytest

from src.core.loop import QueryParams, _run_stop_hooks, query_loop
from src.core.tool import ToolRegistry
from src.core.types import EventType, LoopState, Message, StreamEvent
from src.hooks.engine import Hook, HookEngine, HookEvaluation
from src.llm.client import LLMClient, ModelConfig


class MinLengthHook(Hook):
    name = "min_length"
    min_chars = 50

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        del kwargs
        for msg in reversed(state.messages):
            if msg.role == "assistant":
                if len(msg.text_content.strip()) < self.min_chars:
                    return HookEvaluation(
                        passed=False,
                        feedback="Answer too short. Please expand with more detail.",
                    )
                break
        return HookEvaluation(passed=True)


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
            yield StreamEvent(type=EventType.TEXT_DELTA, data={"text": "Fallback final answer."})
            return

        events = self._script[self._call_index]
        self._call_index += 1
        for event in events:
            yield event


@pytest.mark.asyncio
async def test_run_stop_hooks_without_engine():
    state = LoopState(messages=[], turn_count=1, citations=[])
    params = QueryParams(
        query="test",
        system_prompt="sys",
        tool_registry=ToolRegistry(),
        llm_client=LLMClient(ModelConfig()),
        hook_engine=None,
    )

    should_continue, feedback = await _run_stop_hooks(state, params)

    assert should_continue is False
    assert feedback is None


@pytest.mark.asyncio
async def test_hook_engine_fails_on_short_answer():
    engine = HookEngine()
    engine.register_stop_hook(MinLengthHook())

    state = LoopState(
        messages=[
            Message(role="user", content="Explain Python"),
            Message(role="assistant", content="Too short."),
        ],
        turn_count=1,
        citations=[],
    )

    result = await engine.run_stop_hooks(state)

    assert result.should_continue is True
    assert "too short" in (result.feedback or "").lower()


@pytest.mark.asyncio
async def test_loop_stop_hook_forces_extra_turn():
    engine = HookEngine()
    engine.register_stop_hook(MinLengthHook())

    llm = FakeLLMClient(
        script=[
            [StreamEvent(type=EventType.TEXT_DELTA, data={"text": "Short."})],
            [
                StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={
                        "text": (
                            "Python is a high-level programming language with clear syntax "
                            "and a large standard library."
                        ),
                    },
                ),
            ],
        ]
    )

    params = QueryParams(
        query="What is Python?",
        system_prompt="You are helpful.",
        tool_registry=ToolRegistry(),
        llm_client=llm,
        hook_engine=engine,
        max_turns=5,
    )

    events = [event async for event in query_loop(params)]
    status_messages = [
        event.data.get("message", "") for event in events if event.type == EventType.STATUS
    ]

    assert any("Quality check" in message for message in status_messages)
    done = next(event for event in events if event.type == EventType.DONE)
    assert done.data["turn_count"] == 2
    assert len(done.data["final_answer"]) >= 50
