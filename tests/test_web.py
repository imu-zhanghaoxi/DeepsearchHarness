"""Tests for FastAPI web layer."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.web.router import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "search_web" in data["tools"]
    assert "news_search" in data["tools"]
    assert "academic_search" in data["tools"]
    assert "fetch_url" in data["tools"]
    assert "cite_source" in data["tools"]
    assert "research_plan" in data["tools"]
    assert "deep_read" in data["tools"]
    assert "ask_user" in data["tools"]
    assert "citation_quality" in data["hooks"]
    assert "plan_completeness" in data["hooks"]
    assert data["memory_enabled"] is False
    assert data["skills_enabled"] is False
    assert data["skills_count"] == 0


def test_index_page():
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "DeepsearchHarness" in response.text


def test_login_without_api_key():
    app = create_app()
    client = TestClient(app)
    response = client.post("/api/login", json={"password": "anything"})
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_sessions_crud(tmp_path: Path):
    settings = {
        "sessions": {"base_dir": str(tmp_path / "sessions")},
        "memory": {"enabled": False},
        "skills": {"enabled": False},
    }
    app = create_app(settings=settings)
    client = TestClient(app)

    list_resp = client.get("/api/sessions")
    assert list_resp.status_code == 200
    assert list_resp.json()["sessions"] == []

    missing = client.get("/api/sessions/does-not-exist")
    assert missing.status_code == 404


def test_api_query_requires_body():
    app = create_app()
    client = TestClient(app)
    response = client.post("/api/query", json={})
    assert response.status_code == 400
