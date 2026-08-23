"""Phase 0 foundation tests — backend health endpoint."""

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Health endpoint should return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_returns_status():
    """Health endpoint should include a 'status' field."""
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_health_endpoint_returns_environment():
    """Health endpoint should include the configured environment."""
    response = client.get("/health")
    data = response.json()
    assert "environment" in data
