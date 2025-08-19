"""Tests for cross-encoder reranking functionality."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.services.cross_encoder_reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    """Test cases for cross-encoder reranking."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock cross-encoder model."""
        with patch('src.services.cross_encoder_reranker.CrossEncoder') as mock_ce:
            mock_instance = Mock()
            mock_instance.predict = Mock(return_value=np.array([0.8, 0.6, 0.9, 0.3]))
            mock_ce.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def reranker(self, mock_model):
        """Create reranker instance with mocked model."""
        return CrossEncoderReranker(
            model_name="test-model",
            cache_size=10
        )
    
    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_files",
                    "description": "Search for files in the repository",
                    "parameters": {
                        "properties": {
                            "pattern": {"type": "string"},
                            "path": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "properties": {
                            "filepath": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "properties": {
                            "filepath": {"type": "string"},
                            "content": {"type": "string"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "properties": {
                            "command": {"type": "string"}
                        }
                    }
                }
            }
        ]
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate results from initial search."""
        return [
            {
                "tool_name": "search_files",
                "score": 0.7,
                "description": "Search for files"
            },
            {
                "tool_name": "read_file",
                "score": 0.65,
                "description": "Read file contents"
            },
            {
                "tool_name": "write_file",
                "score": 0.6,
                "description": "Write to file"
            },
            {
                "tool_name": "execute_command",
                "score": 0.4,
                "description": "Execute commands"
            }
        ]
    
    def test_tool_to_text_conversion(self, reranker, sample_tools):
        """Test conversion of tools to text representation."""
        # Test with dict format
        text = reranker._tool_to_text(sample_tools[0])
        assert "search_files" in text
        assert "Search for files" in text
        assert "pattern" in text
        assert "path" in text
        
        # Test with simple dict format
        simple_tool = {
            "tool_name": "test_tool",
            "description": "A test tool"
        }
        text = reranker._tool_to_text(simple_tool)
        assert "test_tool" in text
        assert "A test tool" in text
    
    def test_create_pairs(self, reranker, sample_tools):
        """Test creation of query-tool pairs."""
        query = "I need to search for Python files"
        pairs = reranker._create_pairs(query, sample_tools)
        
        assert len(pairs) == len(sample_tools)
        for i, (q, tool_text) in enumerate(pairs):
            assert q == query
            assert ":" in tool_text  # Format: "name: description"
    
    def test_cache_key_generation(self, reranker):
        """Test cache key generation for query-tool pairs."""
        query = "test query"
        tool_text = "test_tool: A test tool"
        
        key1 = reranker._get_cache_key(query, tool_text)
        key2 = reranker._get_cache_key(query, tool_text)
        key3 = reranker._get_cache_key("different query", tool_text)
        
        assert key1 == key2  # Same inputs produce same key
        assert key1 != key3  # Different inputs produce different keys
    
    @pytest.mark.asyncio
    async def test_rerank_basic(self, reranker, sample_candidates):
        """Test basic reranking functionality."""
        query = "I need to search for files"
        
        # Mock the model prediction
        with patch.object(reranker, '_predict_batch') as mock_predict:
            # Return higher scores for search_files and read_file
            mock_predict.return_value = np.array([0.9, 0.8, 0.3, 0.2])
            
            reranked = await reranker.rerank(
                query=query,
                candidates=sample_candidates,
                top_k=2
            )
            
            assert len(reranked) == 2
            # Check that results are sorted by new scores
            assert reranked[0]["cross_encoder_score"] > reranked[1]["cross_encoder_score"]
    
    @pytest.mark.asyncio
    async def test_rerank_with_original_scores(self, reranker, sample_candidates):
        """Test reranking with original score combination."""
        query = "I need to search for files"
        original_scores = [0.7, 0.65, 0.6, 0.4]
        
        with patch.object(reranker, '_predict_batch') as mock_predict:
            # Return cross-encoder scores
            ce_scores = np.array([0.9, 0.8, 0.3, 0.2])
            mock_predict.return_value = ce_scores
            
            reranked = await reranker.rerank(
                query=query,
                candidates=sample_candidates,
                original_scores=original_scores,
                top_k=3,
                score_combination="weighted"
            )
            
            assert len(reranked) == 3
            # Check that both scores are present
            for result in reranked:
                assert "original_score" in result
                assert "cross_encoder_score" in result
                assert "score" in result  # Combined score
    
    @pytest.mark.asyncio
    async def test_rerank_ce_only(self, reranker, sample_candidates):
        """Test reranking with cross-encoder scores only."""
        query = "I need to write to a file"
        
        with patch.object(reranker, '_predict_batch') as mock_predict:
            # Return scores favoring write_file
            ce_scores = np.array([0.3, 0.4, 0.95, 0.2])
            mock_predict.return_value = ce_scores
            
            reranked = await reranker.rerank(
                query=query,
                candidates=sample_candidates,
                top_k=2,
                score_combination="ce_only"
            )
            
            assert len(reranked) == 2
            # Check that write_file is ranked first
            assert reranked[0]["tool_name"] == "write_file"
            # Score should be the normalized CE score
            assert reranked[0]["score"] == pytest.approx(1 / (1 + np.exp(-0.95)), rel=1e-3)
    
    @pytest.mark.asyncio
    async def test_rerank_with_metadata(self, reranker, sample_candidates):
        """Test reranking candidates that already contain metadata."""
        query = "I need to search for files"
        
        with patch.object(reranker, '_predict_batch') as mock_predict:
            mock_predict.return_value = np.array([0.9, 0.8, 0.3, 0.2])
            
            reranked = await reranker.rerank_with_metadata(
                query=query,
                candidates=sample_candidates,
                top_k=2
            )
            
            assert len(reranked) == 2
            # Original metadata should be preserved
            assert "description" in reranked[0]
    
    def test_predict_batch_with_cache(self, reranker):
        """Test batch prediction with caching."""
        pairs = [
            ("query1", "tool1: description1"),
            ("query2", "tool2: description2"),
            ("query1", "tool1: description1"),  # Duplicate
        ]
        
        with patch.object(reranker.model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([0.8, 0.6])
            
            scores = reranker._predict_batch(pairs)
            
            assert len(scores) == 3
            # Model should only be called for unique pairs
            mock_predict.assert_called_once()
            assert len(mock_predict.call_args[0][0]) == 2  # Only 2 unique pairs
            
            # Check cache was populated
            assert len(reranker._cache) == 2
    
    def test_cache_size_limit(self, reranker):
        """Test that cache respects size limit."""
        reranker.cache_size = 3
        
        # Use _predict_batch to properly add items with cache limit enforcement
        with patch.object(reranker.model, 'predict') as mock_predict:
            # Create pairs that will fill cache beyond limit
            pairs = [(f"query{i}", f"tool{i}: desc{i}") for i in range(5)]
            
            # Mock predictions for all pairs
            mock_predict.return_value = np.array([float(i) for i in range(5)])
            
            # This should add all 5 pairs but respect cache limit
            reranker._predict_batch(pairs)
        
        # Cache should not exceed limit
        assert len(reranker._cache) == 3  # Should be exactly at limit
    
    def test_clear_cache(self, reranker):
        """Test cache clearing."""
        # Populate cache
        reranker._cache["key1"] = 0.5
        reranker._cache["key2"] = 0.7
        
        assert len(reranker._cache) == 2
        
        reranker.clear_cache()
        
        assert len(reranker._cache) == 0
    
    def test_get_cache_stats(self, reranker):
        """Test cache statistics."""
        reranker.cache_size = 100
        
        # Add some cache entries
        for i in range(25):
            reranker._cache[f"key{i}"] = i
        
        stats = reranker.get_cache_stats()
        
        assert stats["cache_size"] == 25
        assert stats["cache_limit"] == 100
        assert stats["cache_usage_percent"] == 25.0
    
    @pytest.mark.asyncio
    async def test_empty_candidates(self, reranker):
        """Test handling of empty candidate list."""
        reranked = await reranker.rerank(
            query="test query",
            candidates=[],
            top_k=10
        )
        
        assert reranked == []
    
    @pytest.mark.asyncio
    async def test_candidates_with_tool_objects(self, reranker, sample_tools):
        """Test reranking with tool objects in candidates."""
        candidates = [
            {"tool": sample_tools[0], "score": 0.7},
            {"tool": sample_tools[1], "score": 0.6}
        ]
        
        with patch.object(reranker, '_predict_batch') as mock_predict:
            mock_predict.return_value = np.array([0.8, 0.9])
            
            reranked = await reranker.rerank(
                query="test query",
                candidates=candidates,
                top_k=2
            )
            
            assert len(reranked) == 2
            assert "cross_encoder_score" in reranked[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])