from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

from src.core.tool import ToolRegistry, ToolUseContext
from src.core.types import (
    ContentBlock,
    EventType,
    LoopState,
    Message,
    StreamEvent,
    ToolResult,
)
from src.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SEARCH_TOOLS = frozenset({"search_web"})
_FETCH_TOOLS = frozenset({"fetch_url"})


@dataclass
class QueryParams:
    query: str
    system_prompt: str
    tool_registry: ToolRegistry
    llm_client: LLMClient
    history: list[Message] = field(default_factory=list)
    max_turns: int = 40
    max_search: int = 30
    max_fetch: int = 30
    hook_engine: object | None = None  # P1
    cache_dir: str = "./cache"
    session_id: str = ""
    rate_limiter: Any = None
    tool_result_preview_chars: int = 2000


async def query_loop(params: QueryParams) -> AsyncGenerator[StreamEvent, None]:
    state = LoopState(
        messages=list(params.history),
        turn_count=0,
        citations=[],
    )

    tool_schemas = params.tool_registry.get_api_schemas()
    concurrent_safe = params.tool_registry.get_concurrent_safe()

    if hasattr(params.llm_client, "reset_response_chain"):
        params.llm_client.reset_response_chain(session_id=params.session_id)

    if not state.messages or state.messages[-1].role != "user":
        state.messages.append(Message(role="user", content=params.query))

    yield StreamEvent(type=EventType.STATUS, data={"message": "Research started"})

    while True:
        state.turn_count += 1

        if state.turn_count > params.max_turns:
            yield StreamEvent(
                type=EventType.STATUS,
                data={
                    "message": (
                        f"Reached maximum turns ({params.max_turns}). "
                        "Synthesizing final answer..."
                    ),
                },
            )
            async for event in _final_answer(state, params):
                yield event
            break

        if state.search_count >= params.max_search and state.fetch_count >= params.max_fetch:
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": "Reached search and fetch limits. Synthesizing final answer..."},
            )
            async for event in _final_answer(state, params):
                yield event
            break

        api_messages = [msg.to_api_dict() for msg in state.messages]
        tool_calls: list[dict] = []
        assistant_text_parts: list[str] = []
        llm_error = False

        try:
            async for event in params.llm_client.stream(
                messages=api_messages,
                system_prompt=params.system_prompt,
                tools=tool_schemas or None,
                session_id=params.session_id,
            ):
                yield event

                if event.type == EventType.TEXT_DELTA:
                    assistant_text_parts.append(event.data.get("text", ""))
                elif event.type == EventType.TOOL_USE:
                    tool_calls.append(event.data)
                elif event.type == EventType.ERROR:
                    logger.error(f"LLM error: {event.data}")
                    llm_error = True
                    break
        except Exception as e:
            logger.error(f"Unexpected error in LLM stream: {e}")
            yield StreamEvent(
                type=EventType.ERROR,
                data={"message": f"LLM stream error: {str(e)}"},
            )
            break

        if llm_error:
            break

        full_text = "".join(assistant_text_parts)
        if tool_calls and full_text:
            blocks = [ContentBlock(type="text", text=full_text)]
            blocks.extend(
                ContentBlock(
                    type="tool_use",
                    tool_use_id=tc["tool_use_id"],
                    tool_name=tc["tool_name"],
                    tool_input=tc["tool_input"],
                )
                for tc in tool_calls
            )
            state.messages.append(Message(role="assistant", content=blocks))
        elif tool_calls:
            state.messages.append(
                Message(
                    role="assistant",
                    content=[
                        ContentBlock(
                            type="tool_use",
                            tool_use_id=tc["tool_use_id"],
                            tool_name=tc["tool_name"],
                            tool_input=tc["tool_input"],
                        )
                        for tc in tool_calls
                    ],
                )
            )
        elif full_text:
            state.messages.append(Message(role="assistant", content=full_text))

        if not tool_calls:
            break

        allowed_tool_calls, skipped_tool_calls = _filter_tool_calls_by_limits(tool_calls, state, params)
        if allowed_tool_calls:
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": f"Executing {len(allowed_tool_calls)} tool(s)..."},
            )

        tool_results = await _execute_tools(
            tool_calls=allowed_tool_calls,
            registry=params.tool_registry,
            state=state,
            params=params,
            concurrent_safe=concurrent_safe,
        )

        for tc in skipped_tool_calls:
            tool_results.append(_limit_reached_result(tc["tool_name"]))
            allowed_tool_calls.append(tc)

        for tc, result in zip(allowed_tool_calls, tool_results):
            result_text = result.data or ""
            streamed_result = result_text[: params.tool_result_preview_chars]

            yield StreamEvent(
                type=EventType.TOOL_RESULT,
                data={
                    "tool_use_id": tc["tool_use_id"],
                    "tool_name": tc["tool_name"],
                    "result": streamed_result,
                    "result_chars": len(result_text),
                    "preview": len(result_text) > len(streamed_result),
                    "is_error": result.is_error,
                    "truncated": result.truncated,
                },
            )

            state.messages.append(
                Message(
                    role="tool",
                    content=result.data,
                    metadata={
                        "tool_call_id": tc["tool_use_id"],
                        "tool_name": tc["tool_name"],
                    },
                )
            )

            tool_name = tc["tool_name"]
            if tool_name in _SEARCH_TOOLS:
                state.search_count += 1
            elif tool_name in _FETCH_TOOLS:
                state.fetch_count += 1

            for citation in result.citations:
                state.citations.append(citation)
                yield StreamEvent(
                    type=EventType.CITATION,
                    data=citation.to_dict(),
                )

    final_answer = _last_assistant_message(state.messages) or ""

    yield StreamEvent(
        type=EventType.STATUS,
        data={
            "message": (
                f"Research complete. {len(state.citations)} sources cited. "
                f"Turns: {state.turn_count}."
            ),
        },
    )

    if hasattr(params.llm_client, "reset_response_chain"):
        params.llm_client.reset_response_chain(session_id=params.session_id)

    yield StreamEvent(
        type=EventType.DONE,
        data={
            "final_answer": final_answer,
            "citations": [c.to_dict() for c in state.citations],
            "turn_count": state.turn_count,
            "compaction_count": state.compaction_count,
        },
    )


def _filter_tool_calls_by_limits(
    tool_calls: list[dict],
    state: LoopState,
    params: QueryParams,
) -> tuple[list[dict], list[dict]]:
    allowed: list[dict] = []
    skipped: list[dict] = []
    pending_search = state.search_count
    pending_fetch = state.fetch_count

    for tc in tool_calls:
        name = tc["tool_name"]
        if name in _SEARCH_TOOLS:
            if pending_search >= params.max_search:
                skipped.append(tc)
                continue
            pending_search += 1
        elif name in _FETCH_TOOLS:
            if pending_fetch >= params.max_fetch:
                skipped.append(tc)
                continue
            pending_fetch += 1
        allowed.append(tc)

    return allowed, skipped


def _limit_reached_result(tool_name: str) -> ToolResult:
    if tool_name in _SEARCH_TOOLS:
        msg = "Search limit reached. You cannot perform more searches."
    elif tool_name in _FETCH_TOOLS:
        msg = "Fetch limit reached. You cannot fetch more pages."
    else:
        msg = f"{tool_name} limit reached."
    return ToolResult(data=msg, is_error=False)


async def _execute_tools(
    tool_calls: list[dict],
    registry: ToolRegistry,
    state: LoopState,
    params: QueryParams,
    concurrent_safe: set[str],
) -> list[ToolResult]:
    context = ToolUseContext(
        session_id=params.session_id,
        turn_count=state.turn_count,
        cache_dir=Path(params.cache_dir),
        extra={
            "loop_state": state,
            "research_query": _extract_research_query(state.messages),
        },
        rate_limiter=params.rate_limiter,
    )

    parallel_indices: list[int] = []
    sequential_indices: list[int] = []

    for i, tc in enumerate(tool_calls):
        if tc["tool_name"] in concurrent_safe:
            parallel_indices.append(i)
        else:
            sequential_indices.append(i)

    results: list[ToolResult] = [
        ToolResult(data="", is_error=True) for _ in range(len(tool_calls))
    ]

    if parallel_indices:
        parallel_results = await asyncio.gather(
            *[
                _execute_single_tool(tool_calls[i], registry, context)
                for i in parallel_indices
            ],
            return_exceptions=True,
        )
        for idx, result in zip(parallel_indices, parallel_results):
            if isinstance(result, Exception):
                logger.error(f"Tool {tool_calls[idx]['tool_name']} failed: {result}")
                results[idx] = ToolResult(
                    data=f"Tool execution failed: {str(result)}",
                    is_error=True,
                )
            else:
                results[idx] = result

    for idx in sequential_indices:
        results[idx] = await _execute_single_tool(tool_calls[idx], registry, context)

    return results


async def _execute_single_tool(
    tc: dict,
    registry: ToolRegistry,
    context: ToolUseContext,
) -> ToolResult:
    tool_name = tc["tool_name"]
    tool_input = tc.get("tool_input", {})

    tool = registry.get(tool_name)
    if tool is None:
        available = ", ".join(t.name for t in registry.all_tools())
        return ToolResult(
            data=f"Error: Unknown tool '{tool_name}'. Available tools: {available}",
            is_error=True,
        )

    validation = tool.validate_input(tool_input)
    if not validation.valid:
        return ToolResult(
            data=(
                f"Invalid input for {tool_name}: {validation.message}. "
                f"You sent: {json.dumps(tool_input)}."
            ),
            is_error=True,
        )

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            return await tool.call(tool_input, context)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str and attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
                continue
            logger.error(f"Tool {tool_name} execution error: {e}", exc_info=True)
            return ToolResult(
                data=f"The {tool_name} service is temporarily unavailable. Please try again later.",
                is_error=False,
            )

    return ToolResult(data=f"Tool {tool_name} failed after retries.", is_error=True)


async def _final_answer(
    state: LoopState,
    params: QueryParams,
) -> AsyncGenerator[StreamEvent, None]:
    state.messages.append(
        Message(
            role="user",
            content=(
                "You have reached the maximum number of tool uses. "
                "Please provide your final answer now based on the "
                "information gathered so far."
            ),
        )
    )

    api_messages = [msg.to_api_dict() for msg in state.messages]
    final_text_parts: list[str] = []

    try:
        async for event in params.llm_client.stream(
            messages=api_messages,
            system_prompt=params.system_prompt,
            tools=None,
            session_id=params.session_id,
        ):
            if event.type == EventType.TEXT_DELTA:
                final_text_parts.append(event.data.get("text", ""))
            yield event
    except Exception as e:
        logger.error(f"Final answer LLM error: {e}")
        yield StreamEvent(
            type=EventType.ERROR,
            data={"message": f"Failed to generate final answer: {str(e)}"},
        )
        return

    final_text = "".join(final_text_parts)
    if final_text.strip():
        state.messages.append(Message(role="assistant", content=final_text))


def _extract_research_query(messages: list[Message]) -> str:
    for msg in messages:
        if msg.role == "user" and not msg.metadata.get("_tag"):
            return msg.text_content
    return ""


def _last_assistant_message(messages: list[Message]) -> str | None:
    for msg in reversed(messages):
        if msg.role == "assistant":
            text = msg.text_content.strip()
            if text:
                return text
    return None
