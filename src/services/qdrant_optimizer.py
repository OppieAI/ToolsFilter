"""Qdrant optimization service for high-performance vector search at scale."""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    SearchParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PointStruct,
    UpdateStatus,
    CollectionInfo
)

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchMode(Enum):
    """Search modes for different accuracy/speed tradeoffs."""
    FAST = "fast"          # Lower accuracy, <10ms
    BALANCED = "balanced"  # Good balance, <50ms  
    ACCURATE = "accurate"  # High accuracy, <100ms
    EXACT = "exact"        # Perfect accuracy, slower


class QdrantOptimizer:
    """
    Optimizes Qdrant for different scale and performance requirements.
    
    Key optimizations:
    1. HNSW parameter tuning based on collection size
    2. Two-stage search within Qdrant
    3. Search parameter optimization
    4. Collection configuration
    5. Batch operations
    """
    
    def __init__(self, client: QdrantClient):
        """Initialize optimizer with Qdrant client."""
        self.client = client
        self.performance_stats = defaultdict(list)
        
    def get_optimized_collection_params(
        self,
        collection_size: int,
        embedding_dim: int = 1536,
        optimize_for: str = "balanced"
    ) -> Tuple[VectorParams, OptimizersConfigDiff]:
        """
        Get optimized collection parameters based on expected size.
        
        Args:
            collection_size: Expected number of vectors
            embedding_dim: Dimension of embeddings
            optimize_for: "speed", "accuracy", or "balanced"
            
        Returns:
            Tuple of (VectorParams, OptimizersConfig)
        """
        # HNSW parameters based on collection size and optimization goal
        if collection_size < 1000:
            # Small collection - can afford higher accuracy
            m = 16
            ef_construct = 200
            full_scan_threshold = 100
        elif collection_size < 10000:
            # Medium collection - balanced settings
            if optimize_for == "speed":
                m = 8
                ef_construct = 100
                full_scan_threshold = 500
            elif optimize_for == "accuracy":
                m = 24
                ef_construct = 400
                full_scan_threshold = 50
            else:  # balanced
                m = 16
                ef_construct = 200
                full_scan_threshold = 200
        elif collection_size < 100000:
            # Large collection - optimize for speed
            if optimize_for == "speed":
                m = 6
                ef_construct = 100
                full_scan_threshold = 1000
            elif optimize_for == "accuracy":
                m = 16
                ef_construct = 300
                full_scan_threshold = 100
            else:  # balanced
                m = 12
                ef_construct = 150
                full_scan_threshold = 500
        else:
            # Very large collection - aggressive optimization
            if optimize_for == "speed":
                m = 4
                ef_construct = 50
                full_scan_threshold = 5000
            else:  # balanced/accuracy
                m = 8
                ef_construct = 100
                full_scan_threshold = 2000
        
        vector_params = VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
            hnsw_config=HnswConfigDiff(
                m=m,
                ef_construct=ef_construct,
                full_scan_threshold=full_scan_threshold,
                max_indexing_threads=4,  # Use multiple threads for indexing
                on_disk=collection_size > 1000000,  # Use disk for very large collections
                payload_m=16,  # Also index payload for filtering
            )
        )
        
        # Optimizer configuration
        optimizer_config = OptimizersConfigDiff(
            deleted_threshold=0.2,  # Vacuum when 20% deleted
            vacuum_min_vector_number=1000,  # Don't vacuum small collections
            default_segment_number=4 if collection_size > 10000 else 2,
            max_segment_size=100000,  # Limit segment size for better parallelism
            memmap_threshold=50000,  # Use memory mapping for large segments
            indexing_threshold=20000,  # Start indexing after this many vectors
            flush_interval_sec=5,  # Flush to disk every 5 seconds
        )
        
        return vector_params, optimizer_config
    
    def get_search_params(self, mode: SearchMode, collection_size: int) -> SearchParams:
        """
        Get optimized search parameters based on mode and collection size.
        
        Args:
            mode: Search mode (FAST, BALANCED, ACCURATE, EXACT)
            collection_size: Current collection size
            
        Returns:
            Optimized SearchParams
        """
        if mode == SearchMode.EXACT:
            return SearchParams(exact=True)
        
        # HNSW ef parameter (higher = more accurate but slower)
        ef_values = {
            SearchMode.FAST: {
                "small": 32,
                "medium": 64,
                "large": 128
            },
            SearchMode.BALANCED: {
                "small": 64,
                "medium": 128,
                "large": 256
            },
            SearchMode.ACCURATE: {
                "small": 128,
                "medium": 256,
                "large": 512
            }
        }
        
        # Determine collection size category
        if collection_size < 1000:
            size_category = "small"
        elif collection_size < 10000:
            size_category = "medium"
        else:
            size_category = "large"
        
        ef = ef_values[mode][size_category]
        
        return SearchParams(
            hnsw_ef=ef,
            exact=False
        )
    
    async def two_stage_search(
        self,
        collection_name: str,
        query_vector: List[float],
        filter_dict: Optional[Dict[str, Any]] = None,
        stage1_limit: int = 1000,
        stage2_limit: int = 10,
        stage1_mode: SearchMode = SearchMode.FAST,
        stage2_mode: SearchMode = SearchMode.ACCURATE
    ) -> List[Dict[str, Any]]:
        """
        Perform two-stage search within Qdrant for better speed/accuracy tradeoff.
        
        Stage 1: Fast approximate search to get candidates
        Stage 2: More accurate search within candidates
        
        Args:
            collection_name: Name of collection to search
            query_vector: Query embedding
            filter_dict: Optional filters
            stage1_limit: Number of candidates to retrieve in stage 1
            stage2_limit: Final number of results
            stage1_mode: Search mode for stage 1 (fast)
            stage2_mode: Search mode for stage 2 (accurate)
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        # Get collection info for optimization
        collection_info = self.client.get_collection(collection_name)
        collection_size = collection_info.points_count
        
        # Build filter if provided
        qdrant_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, list):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value)
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)
        
        # Stage 1: Fast candidate retrieval
        stage1_params = self.get_search_params(stage1_mode, collection_size)
        
        # Debug: Log the filter being used
        if qdrant_filter:
            logger.debug(f"Stage 1 filter: {qdrant_filter}")
            
        stage1_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=qdrant_filter,
            search_params=stage1_params,
            limit=stage1_limit,
            with_payload=False,  # Don't need payload in stage 1
            with_vectors=False   # Don't need vectors in stage 1
        )
        
        stage1_time = time.time() - start_time
        
        if not stage1_results:
            logger.debug(f"Stage 1 returned 0 results with filter: {qdrant_filter}")
            return []
        
        # Get top candidate IDs
        candidate_ids = [hit.id for hit in stage1_results[:min(100, len(stage1_results))]]
        
        # Stage 2: Accurate re-ranking of candidates
        # For stage 2, we need to search within the candidate IDs
        # Since Qdrant doesn't support filtering by point IDs directly in search,
        # we'll retrieve the candidates and re-rank them
        
        # Retrieve candidate points with their payloads
        candidate_points = self.client.retrieve(
            collection_name=collection_name,
            ids=candidate_ids,
            with_payload=True,
            with_vectors=True  # Need vectors for re-scoring
        )
        
        if not candidate_points:
            return []
        
        # Re-score candidates with more accurate parameters
        import numpy as np
        query_vec = np.array(query_vector)
        
        rescored_results = []
        for point in candidate_points:
            # Calculate cosine similarity
            point_vec = np.array(point.vector)
            similarity = np.dot(query_vec, point_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(point_vec))
            
            rescored_results.append({
                "id": point.id,
                "score": float(similarity),
                "payload": point.payload
            })
        
        # Sort by score and take top results
        rescored_results.sort(key=lambda x: x["score"], reverse=True)
        stage2_results = rescored_results[:stage2_limit]
        
        total_time = (time.time() - start_time) * 1000
        
        # Track performance
        self.performance_stats["two_stage_search"].append({
            "total_ms": total_time,
            "stage1_ms": stage1_time * 1000,
            "stage2_ms": (total_time - stage1_time * 1000),
            "candidates": len(candidate_ids),
            "results": len(stage2_results)
        })
        
        logger.debug(
            f"Two-stage search completed in {total_time:.2f}ms "
            f"(Stage1: {stage1_time*1000:.2f}ms, Stage2: {total_time-stage1_time*1000:.2f}ms)"
        )
        
        # Convert to our format
        results = []
        for hit in stage2_results:
            payload = hit.get("payload", {})
            result = {
                "tool_name": payload.get("name", str(hit.get("id", ""))),
                "score": hit.get("score", 0.0),
                **payload
            }
            results.append(result)
        
        return results
    
    async def optimize_existing_collection(
        self,
        collection_name: str,
        target_mode: str = "balanced"
    ) -> bool:
        """
        Optimize an existing collection's configuration.
        
        Args:
            collection_name: Name of collection to optimize
            target_mode: "speed", "accuracy", or "balanced"
            
        Returns:
            True if optimization successful
        """
        try:
            # Get current collection info
            info = self.client.get_collection(collection_name)
            current_vectors = info.points_count
            
            logger.info(f"Optimizing collection '{collection_name}' with {current_vectors} vectors")
            
            # Get optimized parameters
            vector_params, optimizer_config = self.get_optimized_collection_params(
                current_vectors,
                info.config.params.vectors.size,
                target_mode
            )
            
            # Update collection configuration
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=optimizer_config,
                hnsw_config=vector_params.hnsw_config
            )
            
            logger.info(f"Successfully optimized collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize collection: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for operation, timings in self.performance_stats.items():
            if timings:
                times = [t.get("total_ms", 0) for t in timings]
                stats[operation] = {
                    "count": len(timings),
                    "avg_ms": np.mean(times),
                    "p50_ms": np.percentile(times, 50),
                    "p95_ms": np.percentile(times, 95),
                    "p99_ms": np.percentile(times, 99)
                }
        
        return stats
    
    async def bulk_index_with_optimization(
        self,
        collection_name: str,
        tools: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """
        Bulk index tools with optimized batching.
        
        Args:
            collection_name: Collection to index into
            tools: List of tool dictionaries
            embeddings: List of embeddings
            batch_size: Batch size for indexing
            
        Returns:
            Number of tools indexed
        """
        from uuid import uuid4
        
        start_time = time.time()
        total_indexed = 0
        
        # Process in batches for optimal performance
        for i in range(0, len(tools), batch_size):
            batch_tools = tools[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            points = []
            for j, (tool, embedding) in enumerate(zip(batch_tools, batch_embeddings)):
                # Generate proper UUID if tool doesn't have one
                if "id" in tool and tool["id"]:
                    point_id = str(tool["id"])
                else:
                    point_id = str(uuid4())
                
                # Extract tool information
                if tool.get("type") == "function":
                    # Handle both old nested and new flat structure
                    if "function" in tool:
                        function = tool.get("function", {})
                        name = function.get("name", "unknown")
                        description = function.get("description", "")
                    else:
                        name = tool.get("name", "unknown")
                        description = tool.get("description", "")
                else:
                    name = tool.get("name", "unknown")
                    description = tool.get("description", "")
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "name": name,
                            "description": description,
                            "original": tool
                        }
                    )
                )
            
            # Upsert batch
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            total_indexed += len(points)
            logger.debug(f"Indexed batch {i//batch_size + 1}: {len(points)} tools")
        
        elapsed = time.time() - start_time
        logger.info(f"Indexed {total_indexed} tools in {elapsed:.2f}s ({total_indexed/elapsed:.1f} tools/sec)")
        
        return total_indexed


class SearchCache:
    """
    Caching layer for frequently accessed search results.
    
    Uses LRU strategy with TTL for cache entries.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries
        """
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, query_vector: List[float], filters: Optional[Dict] = None) -> str:
        """Create cache key from query and filters."""
        # Use first few dimensions of vector for key (to save memory)
        vector_key = "_".join(str(v)[:6] for v in query_vector[:5])
        filter_key = str(sorted(filters.items())) if filters else "no_filter"
        return f"{vector_key}_{filter_key}"
    
    def get(self, query_vector: List[float], filters: Optional[Dict] = None) -> Optional[List[Dict]]:
        """Get cached results if available and not expired."""
        key = self._make_key(query_vector, filters)
        
        if key in self.cache:
            # Check if expired
            if time.time() - self.access_times[key] < self.ttl_seconds:
                self.hits += 1
                self.access_times[key] = time.time()  # Update access time
                return self.cache[key]
            else:
                # Expired, remove from cache
                del self.cache[key]
                del self.access_times[key]
        
        self.misses += 1
        return None
    
    def set(self, query_vector: List[float], filters: Optional[Dict], results: List[Dict]):
        """Cache search results."""
        key = self._make_key(query_vector, filters)
        
        # Evict oldest entry if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = results
        self.access_times[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0