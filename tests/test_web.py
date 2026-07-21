"""Tests for FastAPI web layer."""

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
    assert "fetch_url" in data["tools"]
    assert "cite_source" in data["tools"]
    assert "research_plan" in data["tools"]
    assert "citation_quality" in data["hooks"]
    assert "plan_completeness" in data["hooks"]


def test_index_page():
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "DeepsearchHarness" in response.text
