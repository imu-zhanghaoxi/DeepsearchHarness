"""Tests for memory retrieval and formatting."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.memory.retrieval import find_relevant_memories, format_memories_for_prompt
from src.memory.store import MemoryEntry, MemoryStore
from src.memory.types import MemoryType


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(base_dir=tmp_path / "memory")


@pytest.mark.asyncio
async def test_find_relevant_memories_returns_empty_when_no_memories(store: MemoryStore):
    result = await find_relevant_memories("quantum computing", store)
    assert result == []


@pytest.mark.asyncio
async def test_find_relevant_memories_uses_side_query(store: MemoryStore):
    await store.save(
        MemoryEntry(
            title="Quantum interest",
            content="User often asks about quantum hardware.",
            memory_type=MemoryType.USER,
        )
    )
    await store.save(
        MemoryEntry(
            title="Cooking preference",
            content="User likes Italian food.",
            memory_type=MemoryType.REFERENCE,
        )
    )

    with patch(
        "src.llm.client.side_query",
        new_callable=AsyncMock,
        return_value='{"selected": ["Quantum interest"]}',
    ):
        result = await find_relevant_memories("latest quantum chip news", store, max_memories=3)

    assert len(result) == 1
    assert result[0].title == "Quantum interest"


def test_format_memories_for_prompt():
    text = format_memories_for_prompt(
        [
            MemoryEntry(
                title="Trusted domain",
                content="Use example.edu for this topic.",
                memory_type=MemoryType.SOURCE_REPUTATION,
            )
        ]
    )

    assert "[source_reputation] Trusted domain" in text
    assert "example.edu" in text
