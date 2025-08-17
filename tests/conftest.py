"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import Mock, AsyncMock

# Configure async test support
pytest_plugins = ["pytest_asyncio"]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = AsyncMock()
    service.embed_text = AsyncMock(return_value=[0.1] * 1024)
    service.embed_conversation = AsyncMock(return_value=[0.1] * 1024)
    service.embed_tool = AsyncMock(return_value=[0.1] * 1024)
    service.embed_batch = AsyncMock(return_value=[[0.1] * 1024])
    return service


@pytest.fixture
def mock_vector_store():
    """Mock vector store service for testing."""
    store = AsyncMock()
    store.search_similar_tools = AsyncMock(return_value=[
        {
            "tool_name": "find",
            "score": 0.95,
            "description": "Find files"
        }
    ])
    store.index_tools_batch = AsyncMock(return_value=["id1", "id2"])
    store.get_tool_by_name = AsyncMock(return_value=None)
    store.get_collection_info = AsyncMock(return_value={
        "vectors_count": 100,
        "indexed_vectors_count": 100
    })
    return store


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "I need to find Python files"},
        {"role": "assistant", "content": "I'll help you find Python files"},
        {"role": "user", "content": "Look in the src directory"}
    ]


@pytest.fixture
def sample_tools():
    """Sample tool definitions."""
    return [
        {
            "type": "function",
            "function": {
                "name": "find",
                "description": "Find files and directories",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "path": {"type": "string"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "Search for patterns in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "file": {"type": "string"}
                    }
                }
            }
        }
    ]