"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock
from src.api.main import app


class TestAPI:
    """Test cases for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @patch("src.api.main.get_vector_store")
    @patch("src.api.main.get_embedding_service")
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "services" in data

    @patch("src.api.main.get_vector_store")
    @patch("src.api.main.get_embedding_service")
    @patch("src.api.main.get_fallback_vector_store")
    @patch("src.api.main.get_fallback_embedding_service")
    def test_tool_filter_endpoint(
        self,
        mock_vector_store,
        mock_embedding_service,
        client,
        sample_messages,
        sample_tools
    ):
        """Test tool filter endpoint."""
        # Setup mocks
        mock_embedding = AsyncMock()
        mock_embedding.embed_conversation = AsyncMock(return_value=[0.1] * 1024)
        mock_embedding.embed_tool = AsyncMock(return_value=[0.1] * 1024)
        mock_embedding_service.return_value = mock_embedding

        mock_store = AsyncMock()
        mock_store.search_similar_tools = AsyncMock(return_value=[
            {"tool_name": "find", "score": 0.95, "description": "Find files"},
            {"tool_name": "grep", "score": 0.85, "description": "Search patterns"}
        ])
        mock_store.get_tool_by_name = AsyncMock(return_value=None)
        mock_store.index_tools_batch = AsyncMock(return_value=["id1", "id2"])
        mock_vector_store.return_value = mock_store

        # Make request
        response = client.post(
            "/api/v1/tools/filter",
            json={
                "messages": sample_messages,
                "available_tools": sample_tools,
                "max_tools": 5,
                "include_reasoning": True
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "recommended_tools" in data
        assert "metadata" in data

        # Check recommended tools
        tools = data["recommended_tools"]
        assert len(tools) <= 5
        assert all("tool_name" in tool for tool in tools)
        assert all("confidence" in tool for tool in tools)

        # Check metadata
        metadata = data["metadata"]
        assert "processing_time_ms" in metadata
        assert "request_id" in metadata

    @patch("src.api.main.get_vector_store")
    @patch("src.api.main.get_embedding_service")
    @patch("src.api.main.get_fallback_vector_store")
    @patch("src.api.main.get_fallback_embedding_service")
    def test_tool_filter_invalid_request(self, client):
        """Test tool filter with invalid request."""
        # Missing messages
        response = client.post(
            "/api/v1/tools/filter",
            json={
                "available_tools": []
            }
        )
        assert response.status_code == 422

        # Empty messages
        response = client.post(
            "/api/v1/tools/filter",
            json={
                "messages": [],
                "available_tools": []
            }
        )
        assert response.status_code == 422

    @patch("src.api.main.get_vector_store")
    @patch("src.api.main.get_embedding_service")
    @patch("src.api.main.get_fallback_vector_store")
    @patch("src.api.main.get_fallback_embedding_service")
    def test_tool_search_endpoint(
        self,
        mock_vector_store,
        mock_embedding_service,
        client
    ):
        """Test tool search endpoint."""
        # Setup mocks
        mock_embedding = AsyncMock()
        mock_embedding.embed_text = AsyncMock(return_value=[0.1] * 1024)
        mock_embedding_service.return_value = mock_embedding

        mock_store = AsyncMock()
        mock_store.search_similar_tools = AsyncMock(return_value=[
            {"tool_name": "find", "score": 0.95}
        ])
        mock_vector_store.return_value = mock_store

        # Make request
        response = client.get(
            "/api/v1/tools/search",
            params={"query": "find files", "limit": 5}
        )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert data["query"] == "find files"

    @patch("src.api.main.get_vector_store")
    @patch("src.api.main.get_embedding_service")
    @patch("src.api.main.get_fallback_vector_store")
    @patch("src.api.main.get_fallback_embedding_service")
    def test_tools_info_endpoint(self, mock_vector_store, client):
        """Test tools info endpoint."""
        # Setup mock
        mock_store = AsyncMock()
        mock_store.get_collection_info = AsyncMock(return_value={
            "vectors_count": 100,
            "indexed_vectors_count": 100
        })
        mock_vector_store.return_value = mock_store

        # Make request
        response = client.get("/api/v1/tools/info")

        assert response.status_code == 200
        data = response.json()
        assert "total_tools" in data
        assert "indexed_tools" in data
