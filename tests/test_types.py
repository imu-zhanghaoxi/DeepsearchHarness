"""Tests for src.core.types — aligned with P0 types.py."""

import json

import pytest

from src.core.types import (
    Citation,
    ContentBlock,
    EventType,
    LoopState,
    Message,
    SourceType,
    StreamEvent,
    ToolResult,
    ValidationResult,
)


class TestCitation:
    def test_to_dict_includes_core_fields(self):
        citation = Citation(
            url="https://example.com",
            title="Example",
            snippet="A snippet",
            source_type=SourceType.WEB,
            relevance_score=0.5,
            cited=True,
        )
        data = citation.to_dict()

        assert data["url"] == "https://example.com"
        assert data["title"] == "Example"
        assert data["snippet"] == "A snippet"
        assert data["source_type"] == "web"
        assert data["relevance_score"] == 0.5
        assert data["cited"] is True
        assert "accessed_at" in data


class TestMessageTextContent:
    def test_string_content(self):
        msg = Message(role="user", content="hello world")
        assert msg.text_content == "hello world"

    def test_blocks_join_text_and_tool_result(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type="text", text="Hello"),
                ContentBlock(type="tool_use", tool_name="search_web"),
                ContentBlock(type="tool_result", content="result data"),
            ],
        )
        assert msg.text_content == "Hello result data"

    def test_ignores_tool_use_blocks(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="tool_use", tool_name="search_web")],
        )
        assert msg.text_content == ""


class TestMessageToApiDict:
    def test_simple_string_message(self):
        msg = Message(role="user", content="What is Python?")
        assert msg.to_api_dict() == {"role": "user", "content": "What is Python?"}

    def test_tool_role_with_string_content(self):
        msg = Message(
            role="tool",
            content='{"results": []}',
            metadata={"tool_call_id": "call_abc"},
        )
        assert msg.to_api_dict() == {
            "role": "tool",
            "content": '{"results": []}',
            "tool_call_id": "call_abc",
        }

    def test_tool_role_missing_tool_call_id_defaults_empty(self):
        msg = Message(role="tool", content="done")
        assert msg.to_api_dict()["tool_call_id"] == ""

    def test_assistant_text_only(self):
        msg = Message(
            role="assistant",
            content=[ContentBlock(type="text", text="Here is the answer.")],
        )
        assert msg.to_api_dict() == {
            "role": "assistant",
            "content": "Here is the answer.",
        }

    def test_assistant_tool_calls(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(
                    type="tool_use",
                    tool_name="search_web",
                    tool_use_id="call_123",
                    tool_input={"query": "python tutorials"},
                ),
            ],
        )
        api = msg.to_api_dict()

        assert api["role"] == "assistant"
        assert api["content"] is None
        assert len(api["tool_calls"]) == 1
        assert api["tool_calls"][0] == {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": json.dumps({"query": "python tutorials"}),
            },
        }

    def test_assistant_text_and_tool_calls(self):
        msg = Message(
            role="assistant",
            content=[
                ContentBlock(type="text", text="Let me search."),
                ContentBlock(
                    type="tool_use",
                    tool_name="fetch_url",
                    tool_use_id="call_456",
                    tool_input={"url": "https://example.com"},
                ),
            ],
        )
        api = msg.to_api_dict()

        assert api["content"] == "Let me search."
        assert len(api["tool_calls"]) == 1
        assert api["tool_calls"][0]["function"]["name"] == "fetch_url"

    def test_non_assistant_blocks_fallback(self):
        msg = Message(
            role="user",
            content=[ContentBlock(type="text", text="Hello")],
        )
        api = msg.to_api_dict()

        assert api["role"] == "user"
        assert api["content"] == [{"type": "text", "text": "Hello"}]


class TestToolResult:
    def test_defaults(self):
        result = ToolResult(data="ok")

        assert result.data == "ok"
        assert result.citations == []
        assert result.truncated is False
        assert result.cached_path is None
        assert result.is_error is False
        assert result.metadata == {}


class TestStreamEvent:
    def test_to_dict(self):
        event = StreamEvent(
            type=EventType.TEXT_DELTA,
            data={"text": "hello"},
        )
        assert event.to_dict() == {
            "type": "text_delta",
            "data": {"text": "hello"},
        }

    @pytest.mark.parametrize(
        "event_type",
        [
            EventType.TEXT_DELTA,
            EventType.TOOL_USE,
            EventType.TOOL_RESULT,
            EventType.CITATION,
            EventType.DONE,
            EventType.ERROR,
        ],
    )
    def test_event_type_values(self, event_type: EventType):
        assert isinstance(event_type.value, str)
        assert StreamEvent(type=event_type).to_dict()["type"] == event_type.value


class TestLoopState:
    def test_defaults(self):
        state = LoopState(messages=[])

        assert state.messages == []
        assert state.turn_count == 0
        assert state.citations == []
        assert state.compaction_count == 0
        assert state.search_count == 0
        assert state.fetch_count == 0


class TestValidationResult:
    def test_valid_without_message(self):
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.message == ""

    def test_invalid_with_message(self):
        result = ValidationResult(valid=False, message="Query is required")
        assert result.valid is False
        assert result.message == "Query is required"
