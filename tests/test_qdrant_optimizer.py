"""Tests for the Qdrant optimizer service."""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from src.services.qdrant_optimizer import QdrantOptimizer, SearchMode, SearchCache


class TestQdrantOptimizer:
    """Test cases for QdrantOptimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.optimizer = QdrantOptimizer(self.mock_client)
    
    def test_get_optimized_collection_params_small(self):
        """Test collection parameters for small collections."""
        vector_params, optimizer_config = self.optimizer.get_optimized_collection_params(
            collection_size=500,
            embedding_dim=1536,
            optimize_for="balanced"
        )
        
        # Check HNSW parameters for small collection
        assert vector_params.size == 1536
        assert vector_params.hnsw_config.m == 16
        assert vector_params.hnsw_config.ef_construct == 200
        assert vector_params.hnsw_config.full_scan_threshold == 100
        
        # Check optimizer config
        assert optimizer_config.default_segment_number == 2
    
    def test_get_optimized_collection_params_medium(self):
        """Test collection parameters for medium collections."""
        vector_params, optimizer_config = self.optimizer.get_optimized_collection_params(
            collection_size=5000,
            embedding_dim=1536,
            optimize_for="speed"
        )
        
        # Check HNSW parameters for medium collection optimized for speed
        assert vector_params.hnsw_config.m == 8
        assert vector_params.hnsw_config.ef_construct == 100
        assert vector_params.hnsw_config.full_scan_threshold == 500
    
    def test_get_optimized_collection_params_large(self):
        """Test collection parameters for large collections."""
        vector_params, optimizer_config = self.optimizer.get_optimized_collection_params(
            collection_size=50000,
            embedding_dim=1536,
            optimize_for="accuracy"
        )
        
        # Check HNSW parameters for large collection optimized for accuracy
        assert vector_params.hnsw_config.m == 16
        assert vector_params.hnsw_config.ef_construct == 300
        assert vector_params.hnsw_config.full_scan_threshold == 100
    
    def test_get_optimized_collection_params_very_large(self):
        """Test collection parameters for very large collections."""
        vector_params, optimizer_config = self.optimizer.get_optimized_collection_params(
            collection_size=500000,
            embedding_dim=1536,
            optimize_for="speed"
        )
        
        # Check HNSW parameters for very large collection
        assert vector_params.hnsw_config.m == 4
        assert vector_params.hnsw_config.ef_construct == 50
        assert vector_params.hnsw_config.full_scan_threshold == 5000
        assert vector_params.hnsw_config.on_disk == False  # Only on disk for 1M+
        
        # Check for 1M+ collection
        vector_params, _ = self.optimizer.get_optimized_collection_params(
            collection_size=1500000,
            embedding_dim=1536,
            optimize_for="speed"
        )
        assert vector_params.hnsw_config.on_disk == True
    
    def test_get_search_params(self):
        """Test search parameter optimization."""
        # Test FAST mode
        params = self.optimizer.get_search_params(SearchMode.FAST, 500)
        assert params.hnsw_ef == 32
        assert params.exact == False
        
        # Test BALANCED mode
        params = self.optimizer.get_search_params(SearchMode.BALANCED, 5000)
        assert params.hnsw_ef == 128
        
        # Test ACCURATE mode
        params = self.optimizer.get_search_params(SearchMode.ACCURATE, 50000)
        assert params.hnsw_ef == 512
        
        # Test EXACT mode
        params = self.optimizer.get_search_params(SearchMode.EXACT, 1000)
        assert params.exact == True
    
    @pytest.mark.asyncio
    async def test_two_stage_search(self):
        """Test two-stage search functionality."""
        # Mock collection info
        mock_collection = MagicMock()
        mock_collection.points_count = 10000
        self.mock_client.get_collection.return_value = mock_collection
        
        # Mock search results
        mock_results = [
            MagicMock(id=f"tool_{i}", score=0.9 - i*0.1, payload={"name": f"tool_{i}"})
            for i in range(5)
        ]
        self.mock_client.search.return_value = mock_results
        
        # Perform two-stage search
        results = await self.optimizer.two_stage_search(
            collection_name="test_collection",
            query_vector=[0.1] * 1536,
            stage1_limit=100,
            stage2_limit=5
        )
        
        # Verify results
        assert len(results) <= 5
        assert self.mock_client.search.call_count == 2  # Stage 1 and Stage 2
        
        # Check performance tracking
        stats = self.optimizer.get_performance_stats()
        assert "two_stage_search" in stats
    
    @pytest.mark.asyncio
    async def test_optimize_existing_collection(self):
        """Test optimizing an existing collection."""
        # Mock collection info
        mock_info = MagicMock()
        mock_info.points_count = 10000
        mock_info.config.params.vectors.size = 1536
        self.mock_client.get_collection.return_value = mock_info
        
        # Mock update_collection
        self.mock_client.update_collection = MagicMock()
        
        # Optimize collection
        success = await self.optimizer.optimize_existing_collection(
            "test_collection",
            target_mode="speed"
        )
        
        assert success == True
        self.mock_client.update_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_bulk_index_with_optimization(self):
        """Test bulk indexing with optimization."""
        # Create test tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"Tool {i} description"
                }
            }
            for i in range(200)
        ]
        
        # Create test embeddings
        embeddings = [[0.1] * 1536 for _ in range(200)]
        
        # Mock upsert
        self.mock_client.upsert = MagicMock()
        
        # Perform bulk indexing
        count = await self.optimizer.bulk_index_with_optimization(
            collection_name="test_collection",
            tools=tools,
            embeddings=embeddings,
            batch_size=50
        )
        
        assert count == 200
        # Should be called 4 times (200 tools / 50 batch size)
        assert self.mock_client.upsert.call_count == 4
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        # Initially empty
        stats = self.optimizer.get_performance_stats()
        assert stats == {}
        
        # Add some mock timings
        self.optimizer.performance_stats["two_stage_search"] = [
            {"total_ms": 50, "stage1_ms": 20, "stage2_ms": 30},
            {"total_ms": 60, "stage1_ms": 25, "stage2_ms": 35},
            {"total_ms": 55, "stage1_ms": 22, "stage2_ms": 33}
        ]
        
        stats = self.optimizer.get_performance_stats()
        assert "two_stage_search" in stats
        assert stats["two_stage_search"]["count"] == 3
        assert stats["two_stage_search"]["avg_ms"] == 55.0
        assert stats["two_stage_search"]["p50_ms"] == 55.0


class TestSearchCache:
    """Test cases for SearchCache."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.cache = SearchCache(max_size=3, ttl_seconds=5)
    
    def test_cache_get_miss(self):
        """Test cache miss."""
        result = self.cache.get([0.1, 0.2], {"filter": "test"})
        assert result is None
        assert self.cache.misses == 1
        assert self.cache.hits == 0
    
    def test_cache_set_and_get(self):
        """Test cache set and get."""
        query = [0.1, 0.2, 0.3]
        filters = {"category": "test"}
        results = [{"tool": "tool1", "score": 0.9}]
        
        # Set cache
        self.cache.set(query, filters, results)
        
        # Get from cache
        cached = self.cache.get(query, filters)
        assert cached == results
        assert self.cache.hits == 1
        assert self.cache.misses == 0
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to max size
        self.cache.set([0.1], None, [{"tool": "tool1"}])
        self.cache.set([0.2], None, [{"tool": "tool2"}])
        self.cache.set([0.3], None, [{"tool": "tool3"}])
        
        assert len(self.cache.cache) == 3
        
        # Add one more - should evict oldest
        self.cache.set([0.4], None, [{"tool": "tool4"}])
        
        assert len(self.cache.cache) == 3
        # First entry should be evicted
        assert self.cache.get([0.1], None) is None
        # Others should still be there
        assert self.cache.get([0.2], None) is not None
    
    def test_cache_expiry(self):
        """Test TTL expiry."""
        import time
        
        # Create cache with 1 second TTL
        cache = SearchCache(max_size=10, ttl_seconds=1)
        
        cache.set([0.1], None, [{"tool": "tool1"}])
        
        # Should be available immediately
        assert cache.get([0.1], None) is not None
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get([0.1], None) is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        self.cache.set([0.1], None, [{"tool": "tool1"}])
        self.cache.get([0.1], None)  # Hit
        self.cache.get([0.2], None)  # Miss
        
        stats = self.cache.get_stats()
        assert stats["size"] == 1
        assert stats["max_size"] == 3
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl_seconds"] == 5
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        self.cache.set([0.1], None, [{"tool": "tool1"}])
        self.cache.set([0.2], None, [{"tool": "tool2"}])
        
        assert len(self.cache.cache) == 2
        
        self.cache.clear()
        
        assert len(self.cache.cache) == 0
        assert self.cache.hits == 0
        assert self.cache.misses == 0