"""Vector store service using Qdrant for similarity search."""

import logging
from typing import List, Dict, Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint
)


from src.core.config import get_settings
from src.services.qdrant_optimizer import QdrantOptimizer, SearchMode, SearchCache

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStoreService:
    """Service for managing vector embeddings in Qdrant."""
    
    def __init__(self, embedding_dimension: Optional[int] = None, model_name: Optional[str] = None, similarity_threshold: Optional[float] = None):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            prefer_grpc=settings.qdrant_prefer_grpc,
            timeout=30
        )
        self.model_name = model_name or settings.primary_embedding_model
        # Include model name in collection to prevent collisions
        model_suffix = self.model_name.replace("/", "_").replace("-", "_")
        self.collection_name = f"{settings.qdrant_collection_name}_{model_suffix}"
        self.embedding_dimension = embedding_dimension or 1536  # Will be set dynamically
        self.similarity_threshold = similarity_threshold or 0.7
        
        # Initialize optimizer and cache
        self.optimizer = QdrantOptimizer(self.client)
        self.search_cache = SearchCache(
            max_size=getattr(settings, 'search_cache_size', 1000),
            ttl_seconds=getattr(settings, 'search_cache_ttl', 300)
        )
        self.use_two_stage_search = getattr(settings, 'enable_two_stage_search', True)
        self.two_stage_threshold = getattr(settings, 'two_stage_threshold', 1000)
        
    async def initialize(self):
        """Initialize vector store and create collections if needed."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                # Get optimized collection parameters
                # Start with expected size of 1000, will re-optimize later
                vector_params, optimizer_config = self.optimizer.get_optimized_collection_params(
                    collection_size=1000,
                    embedding_dim=self.embedding_dimension,
                    optimize_for="balanced"
                )
                
                # Create collection for tools with optimized metadata
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vector_params,
                    optimizers_config=optimizer_config,
                    shard_number=2,
                    replication_factor=1,
                    write_consistency_factor=1
                )
                # Store collection metadata
                import datetime
                collection_metadata = {
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.embedding_dimension,
                    "created_at": datetime.datetime.now().isoformat(),
                    "description": f"Tool embeddings using {self.model_name}"
                }
                # Store metadata as a special point with a proper UUID
                metadata_id = "00000000-0000-0000-0000-000000000001"
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        PointStruct(
                            id=metadata_id,
                            vector=[0.0] * self.embedding_dimension,  # Dummy vector
                            payload={"_type": "metadata", **collection_metadata}
                        )
                    ]
                )
                logger.info(f"Created collection: {self.collection_name} with model: {self.model_name}")
            else:
                # Verify existing collection has same model
                try:
                    metadata_id = "00000000-0000-0000-0000-000000000001"
                    metadata_points = self.client.retrieve(
                        collection_name=self.collection_name,
                        ids=[metadata_id]
                    )
                    if metadata_points:
                        existing_model = metadata_points[0].payload.get("embedding_model")
                        if existing_model and existing_model != self.model_name:
                            logger.warning(
                                f"Collection {self.collection_name} was created with model {existing_model}, "
                                f"but trying to use with {self.model_name}"
                            )
                except Exception:
                    pass  # Metadata point might not exist in older collections
                logger.info(f"Collection {self.collection_name} already exists")
                
            # Create collection for historical patterns (future use)
            patterns_collection = f"{self.collection_name}_patterns"
            if patterns_collection not in collection_names:
                self.client.create_collection(
                    collection_name=patterns_collection,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                # Store metadata for patterns collection too
                import datetime
                patterns_metadata_id = "00000000-0000-0000-0000-000000000002"
                self.client.upsert(
                    collection_name=patterns_collection,
                    points=[
                        PointStruct(
                            id=patterns_metadata_id,
                            vector=[0.0] * self.embedding_dimension,
                            payload={
                                "_type": "metadata",
                                "embedding_model": self.model_name,
                                "embedding_dimension": self.embedding_dimension,
                                "created_at": datetime.datetime.now().isoformat(),
                                "description": f"Usage patterns using {self.model_name}"
                            }
                        )
                    ]
                )
                logger.info(f"Created collection: {patterns_collection}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    async def index_tool(
        self,
        tool_name: str,
        tool_embedding: List[float],
        tool_metadata: Dict[str, Any]
    ) -> str:
        """
        Index a single tool in the vector store.
        
        Args:
            tool_name: Name of the tool
            tool_embedding: Embedding vector
            tool_metadata: Additional metadata about the tool
            
        Returns:
            ID of the indexed point
        """
        point_id = str(uuid4())
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=tool_embedding,
                        payload={
                            "name": tool_name,
                            **tool_metadata
                        }
                    )
                ]
            )
            logger.debug(f"Indexed tool: {tool_name}")
            return point_id
            
        except Exception as e:
            logger.error(f"Failed to index tool {tool_name}: {e}")
            raise
    
    async def index_tools_batch(
        self,
        tools: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Index multiple tools in batch.
        
        Args:
            tools: List of tool definitions
            embeddings: Corresponding embedding vectors
            
        Returns:
            List of indexed point IDs
        """
        if len(tools) != len(embeddings):
            raise ValueError("Number of tools and embeddings must match")
        
        # Use optimizer for large batches
        if len(tools) > 100:
            try:
                indexed_count = await self.optimizer.bulk_index_with_optimization(
                    collection_name=self.collection_name,
                    tools=tools,
                    embeddings=embeddings,
                    batch_size=100
                )
                logger.info(f"Indexed {indexed_count} tools using optimizer")
                
                # After large batch, optimize collection if needed
                collection_info = self.client.get_collection(self.collection_name)
                if collection_info.points_count > 5000:
                    await self.optimizer.optimize_existing_collection(
                        self.collection_name,
                        target_mode="balanced"
                    )
                
                return [str(uuid4()) for _ in range(indexed_count)]
            except Exception as e:
                logger.warning(f"Failed to use optimized bulk indexing: {e}, falling back to standard")
        
        # Standard indexing for smaller batches
        points = []
        point_ids = []
        
        for tool, embedding in zip(tools, embeddings):
            point_id = str(uuid4())
            point_ids.append(point_id)
            
            # Extract tool information
            # Handle both formats: 
            # 1. Flat format: {type: "function", name: "...", description: "...", parameters: {...}}
            # 2. Nested format: {type: "function", function: {name: "...", ...}}
            if tool.get("type") == "function":
                # Check if it's nested format
                if "function" in tool:
                    function = tool["function"]
                    name = function.get("name", "unknown")
                    description = function.get("description", "")
                    parameters = function.get("parameters", {})
                else:
                    # Flat format (more common)
                    name = tool.get("name", "unknown")
                    description = tool.get("description", "")
                    parameters = tool.get("parameters", {})
            else:
                name = tool.get("name", "unknown")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})
            
            # Enhanced payload with more searchable fields
            param_props = parameters.get("properties", {}) if parameters else {}
            param_names = list(param_props.keys()) if param_props else []
            required_params = parameters.get("required", []) if parameters else []
            
            # Tokenize for better text search
            name_tokens = name.lower().replace("_", " ").replace("-", " ").split()
            desc_tokens = description.lower().split() if description else []
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        # Original fields
                        "name": name,  # Store the name as-is
                        "description": description,
                        "parameters": parameters,
                        "category": tool.get("category", "general"),
                        "original": tool,  # Store original tool definition
                        
                        # Enhanced searchable fields
                        "name_lowercase": name.lower(),
                        "name_tokens": name_tokens,
                        "description_tokens": desc_tokens,
                        "param_names": param_names,
                        "required_params": required_params,
                        "param_count": len(param_names),
                        "has_required_params": len(required_params) > 0,
                        
                        # Combined searchable text for full-text search
                        "searchable_text": f"{name} {description} {' '.join(param_names)}".lower()
                    }
                )
            )
        
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} tools in batch")
            return point_ids
            
        except Exception as e:
            logger.error(f"Failed to index tools batch: {e}")
            raise
    
    async def search_similar_tools(
        self,
        query_embedding: List[float],
        limit: int = 10,
        score_threshold: float = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tools based on embedding.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_dict: Optional filters
            
        Returns:
            List of similar tools with scores
        """
        # Try cache first
        cached_results = self.search_cache.get(query_embedding, filter_dict)
        if cached_results is not None:
            logger.debug("Returning cached search results")
            return cached_results[:limit]
        
        # Use two-stage search if enabled and collection is large
        try:
            collection_info = self.client.get_collection(self.collection_name)
            collection_size = collection_info.points_count
            
            if self.use_two_stage_search and collection_size > self.two_stage_threshold:
                # Use optimized two-stage search for large collections
                results = await self.optimizer.two_stage_search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    filter_dict=filter_dict,
                    stage1_limit=1000,
                    stage2_limit=limit,
                    stage1_mode=SearchMode.FAST,
                    stage2_mode=SearchMode.ACCURATE
                )
                
                # Cache the results
                self.search_cache.set(query_embedding, filter_dict, results)
                return results
        except Exception as e:
            logger.warning(f"Failed to use two-stage search, falling back to standard search: {e}")
        
        try:
            # Build filter if provided
            qdrant_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    if isinstance(value, list):
                        # If value is a list, use MatchAny for "in" condition
                        from qdrant_client.models import MatchAny
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchAny(any=value)
                            )
                        )
                    else:
                        # Single value, use exact match
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value)
                            )
                        )
                qdrant_filter = Filter(must=conditions)
            
            # Build filter to exclude metadata points
            metadata_condition = FieldCondition(
                key="_type",
                match=MatchValue(value="metadata")
            )
            
            if qdrant_filter:
                # Add to existing filter
                qdrant_filter.must_not = qdrant_filter.must_not or []
                qdrant_filter.must_not.append(metadata_condition)
            else:
                # Create new filter
                qdrant_filter = Filter(must_not=[metadata_condition])
            
            # Perform search
            # Use provided threshold, or fall back to instance threshold
            # Note: score_threshold=0.0 should be valid (no threshold)
            if score_threshold is not None:
                threshold = score_threshold
            else:
                threshold = self.similarity_threshold
                
            # For score_threshold=0.0, we want ALL results, so don't pass any threshold
            # Only apply threshold if it's explicitly > 0
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": limit,
                "query_filter": qdrant_filter
            }
            
            # Only add score_threshold if it's positive
            if threshold is not None and threshold > 0:
                search_params["score_threshold"] = threshold
                
            results = self.client.search(**search_params)
            
            # Debug: If we have a filter, check if we're getting all expected results
            if filter_dict and "name" in filter_dict:
                expected_names = filter_dict["name"]
                found_names = [r.payload.get("name") for r in results]
                missing = [n for n in expected_names if n not in found_names]
                if missing:
                    logger.warning(f"Qdrant search missed {len(missing)}/{len(expected_names)} tools despite filter: {missing}")
            
            # Format results
            similar_tools = []
            for result in results:
                tool_info = {
                    "tool_name": result.payload.get("name"),
                    "score": result.score,
                    "description": result.payload.get("description"),
                    "category": result.payload.get("category"),
                    "parameters": result.payload.get("parameters"),
                    "original": result.payload.get("original")
                }
                similar_tools.append(tool_info)
            
            return similar_tools
            
        except Exception as e:
            logger.error(f"Failed to search similar tools: {e}")
            raise
    
    async def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information if found
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=tool_name)
                        )
                    ]
                ),
                limit=1
            )
            
            if results[0]:
                point = results[0][0]
                return {
                    "id": point.id,
                    "name": point.payload.get("name"),
                    "description": point.payload.get("description"),
                    "parameters": point.payload.get("parameters"),
                    "category": point.payload.get("category"),
                    "original": point.payload.get("original")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get tool by name {tool_name}: {e}")
            raise
    
    async def delete_tool(self, tool_name: str) -> bool:
        """
        Delete a tool from the vector store.
        
        Args:
            tool_name: Name of the tool to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="name",
                            match=MatchValue(value=tool_name)
                        )
                    ]
                )
            )
            logger.info(f"Deleted tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete tool {tool_name}: {e}")
            return False
    
    async def update_tool_metadata(
        self,
        tool_name: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a tool.
        
        Args:
            tool_name: Name of the tool
            metadata: New metadata to merge
            
        Returns:
            True if updated successfully
        """
        try:
            # Find the tool first
            tool = await self.get_tool_by_name(tool_name)
            if not tool:
                logger.warning(f"Tool not found: {tool_name}")
                return False
            
            # Update payload
            self.client.update_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[tool["id"]]
            )
            
            logger.info(f"Updated metadata for tool: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tool metadata {tool_name}: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            result = {
                "collection_name": self.collection_name,
                "vectors_count": getattr(info, 'vectors_count', 0),
                "indexed_vectors_count": getattr(info, 'indexed_vectors_count', 0),
                "points_count": getattr(info, 'points_count', 0),
                "segments_count": len(getattr(info, 'segments', [])),
                "config": {}
            }
            
            # Try to get metadata
            try:
                metadata_id = "00000000-0000-0000-0000-000000000001"
                metadata_points = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[metadata_id]
                )
                if metadata_points:
                    result["metadata"] = {
                        k: v for k, v in metadata_points[0].payload.items() 
                        if k != "_type"
                    }
            except Exception:
                pass  # Metadata might not exist
                
            return result
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if vector store is healthy."""
        try:
            # Try to get collections
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return False
    
    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their metadata."""
        try:
            collections = self.client.get_collections().collections
            collection_info = []
            
            for col in collections:
                info = {
                    "name": col.name,
                    "vectors_count": getattr(col, 'vectors_count', 0),
                    "points_count": getattr(col, 'points_count', 0)
                }
                
                # Try to get metadata for each collection
                try:
                    # Try both old and new metadata IDs for compatibility
                    metadata_ids = ["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002", "_metadata"]
                    metadata_points = None
                    for mid in metadata_ids:
                        try:
                            metadata_points = self.client.retrieve(
                                collection_name=col.name,
                                ids=[mid]
                            )
                            if metadata_points:
                                break
                        except:
                            continue
                    if metadata_points:
                        info["metadata"] = {
                            k: v for k, v in metadata_points[0].payload.items() 
                            if k != "_type"
                        }
                except Exception:
                    pass  # Metadata might not exist
                
                collection_info.append(info)
            
            return collection_info
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from optimizer and cache."""
        stats = {
            "optimizer": self.optimizer.get_performance_stats(),
            "cache": self.search_cache.get_stats()
        }
        return stats
    
    async def clear_cache(self):
        """Clear the search cache."""
        self.search_cache.clear()
        logger.info("Search cache cleared")
    
    async def optimize_collection(self, target_mode: str = "balanced") -> bool:
        """
        Optimize the current collection configuration.
        
        Args:
            target_mode: "speed", "accuracy", or "balanced"
            
        Returns:
            True if optimization successful
        """
        return await self.optimizer.optimize_existing_collection(
            self.collection_name,
            target_mode
        )
    
    async def close(self):
        """Close the vector store connection."""
        # Qdrant client doesn't require explicit closing
        logger.info("Vector store service closed")