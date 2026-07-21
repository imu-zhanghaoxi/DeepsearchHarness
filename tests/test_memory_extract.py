"""Tests for post-session memory extraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.memory.extract import _parse_memories, extract_memories
from src.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(base_dir=tmp_path / "memory")


def test_parse_memories_handles_code_fence():
    raw = """```json
[{"title": "Topic preference", "content": "User cares about AI safety.", "type": "user"}]
```"""
    parsed = _parse_memories(raw)

    assert len(parsed) == 1
    assert parsed[0]["title"] == "Topic preference"


@pytest.mark.asyncio
async def test_extract_memories_saves_selected_entries(store: MemoryStore):
    response = """[
      {"title": "Reliable source", "content": "arxiv.org worked well.", "type": "source_reputation"}
    ]"""

    with patch(
        "src.llm.client.side_query",
        new_callable=AsyncMock,
        return_value=response,
    ):
        saved = await extract_memories(
            query="What is transformer architecture?",
            final_answer="Transformers use self-attention mechanisms for sequence modeling.",
            plan_findings="",
            store=store,
        )

    assert len(saved) == 1
    loaded = await store.load_all()
    assert len(loaded) == 1
    assert loaded[0].title == "Reliable source"


@pytest.mark.asyncio
async def test_extract_memories_skips_short_sessions(store: MemoryStore):
    saved = await extract_memories(
        query="Hi",
        final_answer="Hello",
        plan_findings="",
        store=store,
    )

    assert saved == []
    assert await store.load_all() == []
