import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


def get_test_client():
    """Create a test client with mocked dependencies."""
    with patch('app.catalog.load_catalog') as mock_load, \
         patch('app.retriever.CatalogRetriever') as mock_retriever, \
         patch('app.agent.SHLAgent') as mock_agent:
        
        mock_load.return_value = []
        mock_retriever_instance = MagicMock()
        mock_retriever.return_value = mock_retriever_instance
        mock_agent_instance = MagicMock()
        mock_agent.return_value = mock_agent_instance
        
        from app.main import app
        return TestClient(app)


def test_health_returns_ok():
    client = get_test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_response_schema():
    client = get_test_client()
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert isinstance(data["status"], str)
