"""Tests for file-based memory store."""

from __future__ import annotations

import pytest

from src.memory.store import MemoryEntry, MemoryStore
from src.memory.types import MemoryType


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    return MemoryStore(base_dir=tmp_path / "memory")


@pytest.mark.asyncio
async def test_save_and_load_memory(store: MemoryStore):
    entry = MemoryEntry(
        title="Preferred sources",
        content="User prefers primary sources and recent papers.",
        memory_type=MemoryType.FEEDBACK,
        tags=["sources"],
    )

    path = await store.save(entry)
    assert path.exists()

    loaded = await store.load_all()
    assert len(loaded) == 1
    assert loaded[0].title == "Preferred sources"
    assert loaded[0].memory_type == MemoryType.FEEDBACK


@pytest.mark.asyncio
async def test_user_profile_is_singleton_file(store: MemoryStore):
    first = MemoryEntry(
        title="User profile",
        content="Researcher in ML.",
        memory_type=MemoryType.USER,
    )
    second = MemoryEntry(
        title="User profile updated",
        content="Researcher in NLP.",
        memory_type=MemoryType.USER,
    )

    await store.save(first)
    await store.save(second)

    loaded = await store.load_all()
    assert len(loaded) == 1
    assert "NLP" in loaded[0].content
    assert (store.base_dir / "user_profile.md").exists()


@pytest.mark.asyncio
async def test_get_headers_and_index(store: MemoryStore):
    await store.save(
        MemoryEntry(
            title="Trusted domain",
            content="example.edu is reliable for this topic.",
            memory_type=MemoryType.SOURCE_REPUTATION,
        )
    )

    headers = await store.get_headers()
    assert len(headers) == 1
    assert headers[0]["title"] == "Trusted domain"
    assert store.index_path.exists()
