"""
Test API endpoint with hybrid search enabled
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio

# Import the app
from src.api.main import app
from src.core.models import ToolFilterRequest, Tool, ToolFunction, ChatMessage


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_request():
    """Create sample request with tools"""
    return {
        "messages": [
            {"role": "user", "content": "I need to analyze cryptocurrency prices and market data"}
        ],
        "available_tools": [
            {
                "type": "function",
                "function": {
                    "name": "crypto_analyzer",
                    "description": "Analyze cryptocurrency prices and market trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {"type": "string", "description": "Crypto symbol"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "stock_analyzer",
                    "description": "Analyze stock market data and trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string", "description": "Stock ticker"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "weather_api",
                    "description": "Get weather information for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Location name"}
                        }
                    }
                }
            }
        ],
        "max_tools": 2
    }


@patch("src.api.main.get_fallback_vector_store")
@patch("src.api.main.get_fallback_embedding_service")
@patch("src.api.main.get_vector_store")
@patch("src.api.main.get_embedding_service")
@patch("src.api.endpoints.settings")
def test_hybrid_search_endpoint(
    mock_settings,
    mock_get_embedding_service,
    mock_get_vector_store,
    mock_get_fallback_embedding,
    mock_get_fallback_vector,
    client,
    sample_request
):
    """Test that hybrid search is used when enabled"""
    
    # Configure mock settings
    mock_settings.enable_hybrid_search = True
    mock_settings.hybrid_search_method = 'weighted'
    mock_settings.semantic_weight = 0.7
    mock_settings.bm25_weight = 0.3
    mock_settings.primary_similarity_threshold = 0.3
    mock_settings.max_tools_to_return = 5
    mock_settings.enable_query_enhancement = False
    mock_settings.enable_api_key = False  # Disable API key for test
    mock_settings.api_key_header = "X-API-Key"
    mock_settings.api_keys_list = []
    
    # Create mock embedding service
    mock_embedding_service = AsyncMock()
    mock_embedding_service.embed_conversation = AsyncMock(return_value=[0.1] * 1536)
    mock_embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 1536 for _ in range(3)])
    mock_embedding_service.model = "test-model"
    mock_get_embedding_service.return_value = mock_embedding_service
    
    # Create mock vector store
    mock_vector_store = AsyncMock()
    mock_vector_store.get_tool_by_name = AsyncMock(return_value=None)
    mock_vector_store.index_tools_batch = AsyncMock(return_value=["id1", "id2", "id3"])
    
    # Mock hybrid_search results
    mock_hybrid_results = [
        {
            "tool_name": "crypto_analyzer",
            "score": 0.75,
            "description": "Analyze cryptocurrency prices",
            "category": "finance"
        },
        {
            "tool_name": "stock_analyzer", 
            "score": 0.45,
            "description": "Analyze stock market",
            "category": "finance"
        }
    ]
    
    # Track if hybrid_search was called
    hybrid_search_called = False
    hybrid_search_args = {}
    
    async def track_hybrid_search(*args, **kwargs):
        nonlocal hybrid_search_called, hybrid_search_args
        hybrid_search_called = True
        hybrid_search_args = kwargs
        return mock_hybrid_results
    
    mock_vector_store.hybrid_search = track_hybrid_search
    mock_get_vector_store.return_value = mock_vector_store
    
    # Set fallback services to None
    mock_get_fallback_embedding.return_value = None
    mock_get_fallback_vector.return_value = None
    
    # Make the request
    response = client.post("/api/v1/tools/filter", json=sample_request)
    
    # Check response
    assert response.status_code == 200
    data = response.json()
    
    # Verify hybrid search was called
    assert hybrid_search_called, "hybrid_search should have been called"
    
    # Check that the right parameters were passed
    assert hybrid_search_args.get('query') == "I need to analyze cryptocurrency prices and market data"
    assert hybrid_search_args.get('method') == 'weighted'
    assert hybrid_search_args.get('score_threshold') == 0.3
    
    # Check response structure
    assert "recommended_tools" in data
    assert "metadata" in data
    
    # Check metadata includes hybrid search info
    metadata = data["metadata"]
    assert metadata.get("search_method") == "hybrid"
    assert metadata.get("hybrid_method") == "weighted"
    assert metadata.get("semantic_weight") == 0.7
    assert metadata.get("bm25_weight") == 0.3
    
    # Check tools returned
    tools = data["recommended_tools"]
    assert len(tools) <= 2  # max_tools = 2
    if tools:
        assert tools[0]["tool_name"] == "crypto_analyzer"
        assert tools[0]["confidence"] == 0.75


@patch("src.api.main.get_fallback_vector_store")
@patch("src.api.main.get_fallback_embedding_service")
@patch("src.api.main.get_vector_store")
@patch("src.api.main.get_embedding_service")
@patch("src.api.endpoints.QueryEnhancer")
@patch("src.api.endpoints.settings")
def test_hybrid_search_disabled(
    mock_settings,
    mock_query_enhancer,
    mock_get_embedding_service,
    mock_get_vector_store,
    mock_get_fallback_embedding,
    mock_get_fallback_vector,
    client,
    sample_request
):
    """Test that query enhancement is used when hybrid search is disabled"""
    
    # Configure mock settings with hybrid search disabled
    mock_settings.enable_hybrid_search = False
    mock_settings.enable_query_enhancement = True
    mock_settings.primary_similarity_threshold = 0.3
    mock_settings.max_tools_to_return = 5
    mock_settings.enable_api_key = False  # Disable API key for test
    mock_settings.api_key_header = "X-API-Key"
    mock_settings.api_keys_list = []
    
    # Create mock embedding service
    mock_embedding_service = AsyncMock()
    mock_embedding_service.embed_queries = AsyncMock(return_value={"query1": [0.1] * 1536})
    mock_embedding_service.embed_batch = AsyncMock(return_value=[[0.1] * 1536 for _ in range(3)])
    mock_embedding_service.model = "test-model"
    mock_get_embedding_service.return_value = mock_embedding_service
    
    # Create mock vector store
    mock_vector_store = AsyncMock()
    mock_vector_store.get_tool_by_name = AsyncMock(return_value=None)
    mock_vector_store.index_tools_batch = AsyncMock(return_value=["id1", "id2", "id3"])
    
    # Track search method calls
    multi_query_called = False
    hybrid_search_called = False
    
    async def mock_search_multi_query(*args, **kwargs):
        nonlocal multi_query_called
        multi_query_called = True
        return []
    
    async def mock_hybrid_search(*args, **kwargs):
        nonlocal hybrid_search_called
        hybrid_search_called = True
        return []
    
    mock_vector_store.search_multi_query = mock_search_multi_query
    mock_vector_store.hybrid_search = mock_hybrid_search
    mock_get_vector_store.return_value = mock_vector_store
    
    # Mock query enhancer
    mock_query_enhancer_instance = MagicMock()
    mock_query_enhancer_instance.enhance_query.return_value = {
        "queries": {"original": "test query"},
        "weights": {"original": 1.0},
        "metadata": {"topics": ["crypto"]}
    }
    mock_query_enhancer.return_value = mock_query_enhancer_instance
    
    # Set fallback services to None
    mock_get_fallback_embedding.return_value = None
    mock_get_fallback_vector.return_value = None
    
    # Make request
    response = client.post("/api/v1/tools/filter", json=sample_request)
    
    # Check response (should succeed even with no tools returned)
    assert response.status_code == 200
    
    # Verify hybrid_search was NOT called
    assert not hybrid_search_called, "hybrid_search should not have been called"
    
    # Verify multi-query search was called instead
    assert multi_query_called, "search_multi_query should have been called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])