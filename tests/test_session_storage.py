"""Tests for session storage."""

from pathlib import Path

import pytest

from src.utils.session_storage import SessionStorage, is_valid_session_id


@pytest.fixture
def storage(tmp_path: Path) -> SessionStorage:
    return SessionStorage(base_dir=tmp_path / "sessions")


def test_is_valid_session_id():
    assert is_valid_session_id("abc-123_xyz")
    assert not is_valid_session_id("../etc/passwd")
    assert not is_valid_session_id("")


def test_save_load_delete_session(storage: SessionStorage):
    session_id = "sess-001"
    storage.save_session(
        session_id,
        {
            "query": "test query",
            "final_answer": "test answer",
            "turn_count": 3,
            "num_citations": 2,
            "citations": [{"url": "https://example.com"}],
        },
    )

    loaded = storage.load_session(session_id)
    assert loaded is not None
    assert loaded["query"] == "test query"
    assert loaded["final_answer"] == "test answer"
    assert loaded["num_citations"] == 2

    sessions = storage.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == session_id

    assert storage.delete_session(session_id) is True
    assert storage.load_session(session_id) is None
    assert storage.delete_session(session_id) is False
