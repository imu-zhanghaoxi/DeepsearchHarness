"""
LLM client wrapper around litellm (Chat Completions only).

Works with any OpenAI-compatible gateway via ``base_url`` + ``api_key``
(DashScope, DeepSeek, Moonshot, vLLM, etc.). No Responses API / reasoning mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

import litellm

from src.core.types import EventType, StreamEvent

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.drop_params = True


def _parse_tool_args(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {"raw": raw}


def _ensure_openai_prefix(model: str) -> str:
    """
    Route custom ``api_base`` gateways through litellm's openai provider.

    Bare ids like ``qwen-plus`` become ``openai/qwen-plus`` so litellm
    sends an OpenAI-compatible Chat Completions request to ``api_base``.
    Already-prefixed names (``openai/…``, ``deepseek/…``, …) are kept.
    """
    if "/" in model:
        return model
    return f"openai/{model}"


def _is_retryable(error: Exception) -> bool:
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    if "connection" in error_type or "connect" in error_str:
        return True
    if "ratelimit" in error_type or "429" in error_str:
        return True
    if "overloaded" in error_str or "529" in error_str:
        return True
    if "internalservererror" in error_type or "500" in error_str:
        return True
    if "timeout" in error_str or "408" in error_str:
        return True
    return False


@dataclass
class ModelConfig:
    default_model: str = "qwen-plus"
    side_query_model: str = "qwen-turbo"
    fallback_model: str = "qwen-turbo"
    max_tokens: int = 4096
    base_url: str = ""
    side_query_base_url: str = ""
    api_key: str = ""
    max_retries: int = 3
    retry_base_delay_ms: int = 500
    stream: bool = True

    @classmethod
    def from_settings(cls, settings_path: str | Path = "config/settings.yaml") -> ModelConfig:
        path = Path(settings_path)
        if not path.exists():
            logger.info(f"Settings file {path} not found, using defaults")
            return cls()

        try:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f) or {}
            llm = data.get("llm", {})
            return cls(
                default_model=llm.get("default_model", cls.default_model),
                side_query_model=llm.get("side_query_model", cls.side_query_model),
                fallback_model=llm.get("fallback_model", cls.fallback_model),
                max_tokens=int(llm.get("max_tokens", cls.max_tokens)),
                base_url=llm.get("base_url", "") or "",
                side_query_base_url=llm.get("side_query_base_url", "") or "",
                api_key=llm.get("api_key", "") or "",
                max_retries=int(llm.get("max_retries", cls.max_retries)),
                retry_base_delay_ms=int(llm.get("retry_base_delay_ms", cls.retry_base_delay_ms)),
                stream=bool(llm.get("stream", cls.stream)),
            )
        except Exception as e:
            logger.warning(f"Failed to load settings from {path}: {e}, using defaults")
            return cls()

    def resolve_api_key(self) -> str:
        return (
            self.api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
            or ""
        )


class LLMClient:
    """litellm Chat Completions client with streaming tool calls and retries."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

    def reset_response_chain(self, session_id: str = "") -> None:
        return None

    @property
    def uses_responses_api(self) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    def _completion_kwargs(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        use_stream: bool,
        base_url: str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": _ensure_openai_prefix(model),
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if use_stream:
            kwargs["stream"] = True
        if tools:
            kwargs["tools"] = tools
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice

        url = base_url if base_url is not None else self.config.base_url
        if url:
            kwargs["api_base"] = url

        api_key = self.config.resolve_api_key()
        if api_key:
            kwargs["api_key"] = api_key

        return kwargs

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        session_id: str = "",
        tool_choice: str | dict | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        del session_id
        target_model = model or self.config.default_model
        target_max_tokens = max_tokens or self.config.max_tokens
        api_messages = [{"role": "system", "content": system_prompt}] + messages
        fallback_model = self.config.fallback_model

        last_error: Exception | None = None

        for attempt in range(1, self.config.max_retries + 2):
            try:
                async for event in self._complete(
                    model=target_model,
                    messages=api_messages,
                    max_tokens=target_max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    use_stream=self.config.stream,
                ):
                    yield event
                return
            except Exception as e:
                last_error = e
                logger.error(
                    f"LLM call failed with {target_model} "
                    f"(attempt {attempt}/{self.config.max_retries + 1}): {e}"
                )
                if not _is_retryable(e) or attempt > self.config.max_retries:
                    break

                delay = (self.config.retry_base_delay_ms / 1000) * (2 ** (attempt - 1))
                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{self.config.max_retries})..."
                )
                yield StreamEvent(
                    type=EventType.STATUS,
                    data={
                        "message": (
                            f"API error, retrying in {delay:.0f}s... "
                            f"(attempt {attempt}/{self.config.max_retries})"
                        ),
                    },
                )
                await asyncio.sleep(delay)

        if last_error and target_model != fallback_model:
            logger.info(f"Falling back to {fallback_model}")
            yield StreamEvent(
                type=EventType.STATUS,
                data={"message": f"Switched to fallback model: {fallback_model}"},
            )
            async for event in self.stream(
                messages=messages,
                system_prompt=system_prompt,
                tools=tools,
                model=fallback_model,
                max_tokens=max_tokens,
                tool_choice=tool_choice,
            ):
                yield event
        elif last_error:
            yield StreamEvent(
                type=EventType.ERROR,
                data={"message": f"LLM call failed: {str(last_error)}"},
            )

    async def _complete(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int,
        tools: list[dict] | None,
        tool_choice: str | dict | None,
        use_stream: bool,
    ) -> AsyncGenerator[StreamEvent, None]:
        kwargs = self._completion_kwargs(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            use_stream=use_stream,
        )
        response = await litellm.acompletion(**kwargs)

        if not use_stream:
            msg = response.choices[0].message
            if getattr(msg, "content", None):
                yield StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": msg.content},
                )
            for tc in getattr(msg, "tool_calls", None) or []:
                yield StreamEvent(
                    type=EventType.TOOL_USE,
                    data={
                        "tool_use_id": tc.id,
                        "tool_name": tc.function.name,
                        "tool_input": _parse_tool_args(tc.function.arguments or ""),
                    },
                )
            return

        tool_call_buffers: dict[int, dict] = {}
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                yield StreamEvent(
                    type=EventType.TEXT_DELTA,
                    data={"text": delta.content},
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    if idx not in tool_call_buffers:
                        tool_call_buffers[idx] = {
                            "id": tc.id or f"call_{idx}",
                            "name": "",
                            "args": "",
                        }
                    buf = tool_call_buffers[idx]
                    if tc.id:
                        buf["id"] = tc.id
                    if tc.function:
                        if tc.function.name and not buf["name"]:
                            buf["name"] = tc.function.name
                        if tc.function.arguments:
                            buf["args"] += tc.function.arguments

        for idx in sorted(tool_call_buffers):
            buf = tool_call_buffers[idx]
            yield StreamEvent(
                type=EventType.TOOL_USE,
                data={
                    "tool_use_id": buf["id"],
                    "tool_name": buf["name"],
                    "tool_input": _parse_tool_args(buf["args"]),
                },
            )


async def side_query(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 512,
    output_schema: dict | None = None,
    config: ModelConfig | None = None,
) -> str:
    """Non-streaming helper for ranking / memory / quality checks."""
    cfg = config or _shared_config or ModelConfig()
    target_model = model or cfg.side_query_model
    base_url = cfg.side_query_base_url or cfg.base_url or None

    messages: list[dict] = [{"role": "user", "content": prompt}]
    if system:
        messages = [{"role": "system", "content": system}] + messages

    kwargs: dict[str, Any] = {
        "model": _ensure_openai_prefix(target_model),
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if base_url:
        kwargs["api_base"] = base_url
    api_key = cfg.resolve_api_key()
    if api_key:
        kwargs["api_key"] = api_key

    if output_schema:
        schema_hint = json.dumps(output_schema, ensure_ascii=False)
        hint = "\n\nRespond with a single JSON object matching this schema:\n" + schema_hint
        if messages[0]["role"] == "system":
            messages[0]["content"] += hint
        else:
            messages.insert(0, {"role": "system", "content": hint.lstrip()})
            kwargs["messages"] = messages
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await litellm.acompletion(**kwargs)
        return response.choices[0].message.content or ""
    except Exception as e:
        if output_schema and "response_format" in kwargs:
            kwargs.pop("response_format", None)
            try:
                response = await litellm.acompletion(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e2:
                logger.warning(f"Side query failed (after format fallback): {e2}")
                return ""
        logger.warning(f"Side query failed: {e}")
        return ""


_shared_config: ModelConfig | None = None


def set_shared_config(config: ModelConfig) -> None:
    global _shared_config
    _shared_config = config
