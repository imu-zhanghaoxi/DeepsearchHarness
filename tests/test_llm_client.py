"""Tests for src.llm.client (litellm wrapper)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.types import EventType
from src.llm.client import (
    LLMClient,
    ModelConfig,
    _ensure_openai_prefix,
    _is_retryable,
    _parse_tool_args,
    set_shared_config,
    side_query,
)


class TestParseToolArgs:
    def test_empty_string(self):
        assert _parse_tool_args("") == {}

    def test_valid_json(self):
        assert _parse_tool_args('{"query": "hello"}') == {"query": "hello"}

    def test_invalid_json_wraps_raw(self):
        assert _parse_tool_args("not-json") == {"raw": "not-json"}


class TestEnsureOpenaiPrefix:
    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("qwen-plus", "openai/qwen-plus"),
            ("deepseek-chat", "openai/deepseek-chat"),
            ("openai/gpt-4o", "openai/gpt-4o"),
            ("deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ],
    )
    def test_prefix_rules(self, model: str, expected: str):
        assert _ensure_openai_prefix(model) == expected


class TestIsRetryable:
    def test_connection_error(self):
        assert _is_retryable(ConnectionError("connection reset")) is True

    def test_rate_limit_in_message(self):
        assert _is_retryable(Exception("HTTP 429 Too Many Requests")) is True

    def test_server_error(self):
        assert _is_retryable(Exception("500 Internal Server Error")) is True

    def test_auth_error_not_retryable(self):
        assert _is_retryable(Exception("401 Unauthorized")) is False


class TestModelConfig:
    def test_from_settings_missing_file_uses_defaults(self, tmp_path):
        cfg = ModelConfig.from_settings(tmp_path / "missing.yaml")
        assert cfg.default_model == "qwen-plus"
        assert cfg.max_retries == 3

    def test_from_settings_loads_llm_section(self, tmp_path):
        settings = tmp_path / "settings.yaml"
        settings.write_text(
            """
llm:
  default_model: custom-model
  max_tokens: 2048
  base_url: https://example.com/v1
  max_retries: 1
  stream: false
""".strip()
        )
        cfg = ModelConfig.from_settings(settings)
        assert cfg.default_model == "custom-model"
        assert cfg.max_tokens == 2048
        assert cfg.base_url == "https://example.com/v1"
        assert cfg.max_retries == 1
        assert cfg.stream is False

    def test_resolve_api_key_prefers_config(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        cfg = ModelConfig(api_key="config-key")
        assert cfg.resolve_api_key() == "config-key"

    def test_resolve_api_key_falls_back_to_env(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        cfg = ModelConfig(api_key="")
        assert cfg.resolve_api_key() == "env-key"


class TestLLMClientCompletionKwargs:
    def test_adds_openai_prefix_api_base_and_key(self):
        client = LLMClient(
            ModelConfig(
                base_url="https://gateway.example/v1",
                api_key="secret",
            )
        )
        kwargs = client._completion_kwargs(
            model="qwen-plus",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=128,
            tools=None,
            tool_choice=None,
            use_stream=False,
        )
        assert kwargs["model"] == "openai/qwen-plus"
        assert kwargs["api_base"] == "https://gateway.example/v1"
        assert kwargs["api_key"] == "secret"
        assert "stream" not in kwargs

    def test_stream_and_tools_forwarded(self):
        client = LLMClient()
        tools = [{"type": "function", "function": {"name": "search_web"}}]
        kwargs = client._completion_kwargs(
            model="deepseek/deepseek-chat",
            messages=[],
            max_tokens=64,
            tools=tools,
            tool_choice="auto",
            use_stream=True,
        )
        assert kwargs["model"] == "deepseek/deepseek-chat"
        assert kwargs["stream"] is True
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"


def _non_stream_response(*, content: str = "hello", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


async def _stream_chunks(chunks):
    for chunk in chunks:
        yield chunk


class TestLLMClientComplete:
    @pytest.mark.asyncio
    async def test_non_stream_text_and_tool_calls(self):
        response = _non_stream_response(
            content="Let me search.",
            tool_calls=[
                _tool_call("call_1", "search_web", '{"query": "python"}'),
            ],
        )
        client = LLMClient(ModelConfig(stream=False))

        with patch("src.llm.client.litellm.acompletion", new=AsyncMock(return_value=response)):
            events = [
                event
                async for event in client._complete(
                    model="qwen-plus",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=128,
                    tools=None,
                    tool_choice=None,
                    use_stream=False,
                )
            ]

        assert [e.type for e in events] == [EventType.TEXT_DELTA, EventType.TOOL_USE]
        assert events[0].data["text"] == "Let me search."
        assert events[1].data["tool_name"] == "search_web"
        assert events[1].data["tool_input"] == {"query": "python"}

    @pytest.mark.asyncio
    async def test_stream_reassembles_tool_call_fragments(self):
        chunks = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="Searching",
                            tool_calls=None,
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_9",
                                    function=SimpleNamespace(
                                        name="search_web",
                                        arguments='{"query":',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(
                                        name=None,
                                        arguments=' "ai"}',
                                    ),
                                )
                            ],
                        )
                    )
                ]
            ),
        ]
        client = LLMClient(ModelConfig(stream=True))

        with patch(
            "src.llm.client.litellm.acompletion",
            new=AsyncMock(return_value=_stream_chunks(chunks)),
        ):
            events = [
                event
                async for event in client._complete(
                    model="qwen-plus",
                    messages=[],
                    max_tokens=128,
                    tools=None,
                    tool_choice=None,
                    use_stream=True,
                )
            ]

        assert [e.type for e in events] == [EventType.TEXT_DELTA, EventType.TOOL_USE]
        assert events[0].data["text"] == "Searching"
        assert events[1].data["tool_use_id"] == "call_9"
        assert events[1].data["tool_input"] == {"query": "ai"}


class TestLLMClientStream:
    @pytest.mark.asyncio
    async def test_stream_success_yields_events(self):
        response = _non_stream_response(content="final answer")
        client = LLMClient(ModelConfig(stream=False, default_model="qwen-plus"))

        with patch("src.llm.client.litellm.acompletion", new=AsyncMock(return_value=response)):
            events = [
                event
                async for event in client.stream(
                    messages=[{"role": "user", "content": "hello"}],
                    system_prompt="You are helpful.",
                )
            ]

        assert len(events) == 1
        assert events[0].type == EventType.TEXT_DELTA
        assert events[0].data["text"] == "final answer"

    @pytest.mark.asyncio
    async def test_stream_retries_then_succeeds(self, monkeypatch):
        response = _non_stream_response(content="ok")
        mock_completion = AsyncMock(
            side_effect=[Exception("429 rate limit"), response],
        )
        sleep_calls: list[float] = []

        async def fast_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        monkeypatch.setattr(asyncio, "sleep", fast_sleep)

        client = LLMClient(
            ModelConfig(
                stream=False,
                default_model="qwen-plus",
                fallback_model="qwen-plus",
                max_retries=2,
                retry_base_delay_ms=500,
            )
        )

        with patch("src.llm.client.litellm.acompletion", new=mock_completion):
            events = [
                event
                async for event in client.stream(
                    messages=[{"role": "user", "content": "hi"}],
                    system_prompt="sys",
                )
            ]

        assert mock_completion.await_count == 2
        assert sleep_calls == [0.5]
        assert events[0].type == EventType.STATUS
        assert events[1].type == EventType.TEXT_DELTA

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_secondary_model(self):
        ok_response = _non_stream_response(content="fallback answer")
        mock_completion = AsyncMock(
            side_effect=[
                Exception("401 unauthorized"),
                ok_response,
            ],
        )
        client = LLMClient(
            ModelConfig(
                stream=False,
                default_model="qwen-plus",
                fallback_model="qwen-turbo",
            )
        )

        with patch("src.llm.client.litellm.acompletion", new=mock_completion):
            events = [
                event
                async for event in client.stream(
                    messages=[{"role": "user", "content": "hi"}],
                    system_prompt="sys",
                )
            ]

        assert mock_completion.await_count == 2
        assert events[0].type == EventType.STATUS
        assert "fallback" in events[0].data["message"].lower()
        assert events[1].type == EventType.TEXT_DELTA
        assert events[1].data["text"] == "fallback answer"

    @pytest.mark.asyncio
    async def test_stream_emits_error_when_all_attempts_fail(self):
        mock_completion = AsyncMock(side_effect=Exception("401 unauthorized"))
        client = LLMClient(
            ModelConfig(
                stream=False,
                default_model="qwen-plus",
                fallback_model="qwen-plus",
            )
        )

        with patch("src.llm.client.litellm.acompletion", new=mock_completion):
            events = [
                event
                async for event in client.stream(
                    messages=[{"role": "user", "content": "hi"}],
                    system_prompt="sys",
                )
            ]

        assert len(events) == 1
        assert events[0].type == EventType.ERROR
        assert "401" in events[0].data["message"]


class TestSideQuery:
    @pytest.mark.asyncio
    async def test_returns_message_content(self):
        response = _non_stream_response(content='{"score": 1}')
        cfg = ModelConfig(api_key="k", base_url="https://example.com/v1")

        with patch("src.llm.client.litellm.acompletion", new=AsyncMock(return_value=response)) as mock:
            result = await side_query(
                "rate this",
                system="be concise",
                config=cfg,
                output_schema={"score": "number"},
            )

        assert result == '{"score": 1}'
        kwargs = mock.await_args.kwargs
        assert kwargs["model"] == "openai/qwen-turbo"
        assert kwargs["api_base"] == "https://example.com/v1"
        assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_returns_empty_string_on_failure(self):
        with patch(
            "src.llm.client.litellm.acompletion",
            new=AsyncMock(side_effect=Exception("boom")),
        ):
            result = await side_query("x", config=ModelConfig())

        assert result == ""


class TestSetSharedConfig:
    def test_set_shared_config_used_by_side_query(self):
        cfg = ModelConfig(side_query_model="custom-side")
        set_shared_config(cfg)
        assert cfg.side_query_model == "custom-side"
