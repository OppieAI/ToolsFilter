"""Search service that orchestrates different search strategies."""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from src.core.config import get_settings
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.bm25_ranker import BM25Ranker, HybridScorer
from src.services.cross_encoder_reranker import CrossEncoderReranker
from src.core.models import Tool

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchStrategy(Enum):
    """Available search strategies."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CROSS_ENCODER = "cross_encoder"
    HYBRID_CROSS_ENCODER = "hybrid_cross_encoder"


class SearchService:
    """
    Service that orchestrates different search strategies.
    
    This service acts as the main interface for tool search operations,
    coordinating between vector store (Qdrant), BM25 ranker, and cross-encoder.
    
    Supports dependency injection for testing while providing sensible defaults
    for production use.
    """
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        bm25_ranker: Optional[BM25Ranker] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        hybrid_scorer: Optional[HybridScorer] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize search service with optional dependency injection.
        
        Args:
            vector_store: Vector store service for semantic search (required)
            embedding_service: Embedding service for generating embeddings (required)
            bm25_ranker: Optional BM25 ranker instance (created from config if None)
            cross_encoder: Optional cross-encoder instance (created from config if None)
            hybrid_scorer: Optional hybrid scorer instance (created from config if None)
            config: Optional configuration object (uses global settings if None)
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.config = config or settings
        
        # BM25 components - use injected or create from config
        self.enable_bm25 = getattr(self.config, 'enable_hybrid_search', True)
        self.bm25_ranker = bm25_ranker
        self.hybrid_scorer = hybrid_scorer
        
        if self.enable_bm25 and self.bm25_ranker is None:
            self.bm25_ranker = self._create_bm25_ranker()
            
        if self.enable_bm25 and self.hybrid_scorer is None:
            self.hybrid_scorer = self._create_hybrid_scorer()
            
        if self.enable_bm25:
            logger.info("BM25 hybrid search enabled")
        
        # Cross-encoder - use injected or create from config
        self.enable_cross_encoder = getattr(self.config, 'enable_cross_encoder', True)
        self.cross_encoder = cross_encoder
        
        if self.enable_cross_encoder and self.cross_encoder is None:
            self.cross_encoder = self._create_cross_encoder()
            
        if self.enable_cross_encoder and self.cross_encoder:
            logger.info("Cross-encoder reranking enabled")
    
    def _create_bm25_ranker(self) -> BM25Ranker:
        """Factory method to create BM25 ranker from config."""
        return BM25Ranker(
            variant=getattr(self.config, 'bm25_variant', 'okapi'),
            k1=getattr(self.config, 'bm25_k1', 1.5),
            b=getattr(self.config, 'bm25_b', 0.75)
        )
    
    def _create_hybrid_scorer(self) -> HybridScorer:
        """Factory method to create hybrid scorer from config."""
        if not self.bm25_ranker:
            raise ValueError("BM25 ranker required for hybrid scorer")
            
        return HybridScorer(
            bm25_ranker=self.bm25_ranker,
            semantic_weight=getattr(self.config, 'semantic_weight', 0.7),
            bm25_weight=getattr(self.config, 'bm25_weight', 0.3)
        )
    
    def _create_cross_encoder(self) -> Optional[CrossEncoderReranker]:
        """Factory method to create cross-encoder from config."""
        try:
            return CrossEncoderReranker(
                model_name=getattr(self.config, 'cross_encoder_model', None),
                cache_size=getattr(self.config, 'cross_encoder_cache_size', 1000)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize cross-encoder: {e}")
            self.enable_cross_encoder = False
            return None
    
    async def semantic_search(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        query_embedding: Optional[List[float]] = None,
        available_tools: Optional[List[Any]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure semantic search using vector embeddings.
        
        Args:
            query: Text query (will be embedded)
            messages: Conversation messages (will be embedded)
            query_embedding: Pre-computed query embedding (optional)
            available_tools: Optional list of tools to filter by
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of tools ranked by semantic similarity
        """
        # Generate embedding if not provided
        if query_embedding is None:
            if messages:
                query_embedding = await self.embedding_service.embed_conversation(messages)
            elif query:
                query_embedding = await self.embedding_service.embed_text(query)
            else:
                raise ValueError("Must provide either query, messages, or query_embedding")
        # Build filter if available_tools provided
        filter_dict = None
        if available_tools:
            tool_names = self._extract_tool_names(available_tools)
            if tool_names:
                filter_dict = {"name": tool_names}
        
        # Perform semantic search
        results = await self.vector_store.search_similar_tools(
            query_embedding=query_embedding,
            filter_dict=filter_dict,
            limit=limit,
            score_threshold=score_threshold
        )
        
        logger.info(f"Semantic search returned {len(results)} results")
        return results
    
    async def hybrid_search(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        available_tools: List[Any] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        method: Optional[str] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid semantic + BM25 search.
        
        THIS IS THE EXACT OPTIMIZED IMPLEMENTATION FROM vector_store.py
        
        Args:
            query: Search query text (for BM25 and embedding if not provided)
            messages: Conversation messages (for extracting query and embedding)
            available_tools: List of available tools to search within
            query_embedding: Pre-computed query embedding (optional)
            limit: Maximum number of results
            method: Hybrid method ('weighted', 'rrf', or None for config default)
            score_threshold: Minimum score threshold (None uses default)
            
        Returns:
            List of tools ranked by hybrid score
        """
        if not self.enable_bm25:
            logger.warning("BM25 not enabled, falling back to semantic search")
            return await self.semantic_search(
                query=query, messages=messages, query_embedding=query_embedding,
                available_tools=available_tools, limit=limit, score_threshold=score_threshold
            )
        
        if not available_tools:
            return []
        
        # Extract query text for BM25 if not provided
        if query is None and messages:
            query = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if query is None:
            raise ValueError("Must provide either query or messages for hybrid search")
        
        # Generate embedding if not provided
        if query_embedding is None:
            if messages:
                query_embedding = await self.embedding_service.embed_conversation(messages)
            else:
                query_embedding = await self.embedding_service.embed_text(query)
        
        method = method or getattr(self.config, 'hybrid_search_method', 'weighted')
        
        # Extract tool names for filtering
        tool_names = []
        for tool in available_tools:
            if hasattr(tool, 'function'):
                tool_names.append(tool.function.name)
            elif isinstance(tool, dict) and 'function' in tool:
                tool_names.append(tool['function']['name'])
            else:
                logger.warning(f"Tool without function name: {tool}")
        
        # Step 1: Semantic search via Qdrant
        # Get all available tools to ensure complete ranking
        # Use a high limit to ensure we get all tools
        
        # Debug: Log what we're searching for
        logger.info(f"Hybrid search: Searching for {len(tool_names)} tools: {tool_names}")
        
        semantic_results = await self.vector_store.search_similar_tools(
            query_embedding=query_embedding,
            filter_dict={"name": tool_names} if tool_names else None,
            limit=max(len(available_tools) * 2, 100),  # Higher limit to ensure we get all
            score_threshold=0.0  # No threshold for hybrid
        )
        
        # Debug: Log what we found
        found_tools = [r['tool_name'] for r in semantic_results]
        missing_tools = [name for name in tool_names if name not in found_tools]
        if missing_tools:
            logger.warning(f"Semantic search missed {len(missing_tools)} tools: {missing_tools}")
        
        # Log semantic scores for debugging
        logger.debug(f"Semantic search results ({len(semantic_results)} tools):")
        for r in semantic_results:
            logger.debug(f"  {r['tool_name']}: {r['score']:.3f}")
        
        # Step 2: BM25 scoring
        bm25_scores = self.bm25_ranker.score_tools(query, available_tools)
        
        # Step 3: Merge scores
        if method == "rrf":
            merged_results = self.hybrid_scorer.merge_scores_rrf(
                semantic_results,
                bm25_scores,
                k=60
            )
        else:
            # Default to weighted method
            merged_results = self.hybrid_scorer.merge_scores_weighted(
                semantic_results,
                bm25_scores,
                available_tools
            )
        
        # Step 4: Apply threshold and limit
        # Use provided threshold, or fall back to default
        threshold = score_threshold if score_threshold is not None else self.vector_store.similarity_threshold
        filtered = [r for r in merged_results if r['score'] >= threshold]
        
        # Log hybrid search statistics
        logger.debug(f"Hybrid search: {len(semantic_results)} semantic, "
                    f"{sum(1 for s in bm25_scores.values() if s > 0)} BM25 matches, "
                    f"{len(filtered)} after threshold")
        
        return filtered[:limit]
    
    async def search_multi_query(
        self,
        query_embeddings: Dict[str, List[float]],
        weights: Dict[str, float],
        limit: int = 10,
        score_threshold: float = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar tools using multiple query embeddings with weighted aggregation.
        
        THIS IS THE EXACT OPTIMIZED IMPLEMENTATION FROM vector_store.py
        
        Args:
            query_embeddings: Dictionary of query names to embedding vectors
            weights: Dictionary of query names to weights (should sum to 1.0)
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_dict: Optional filters
            
        Returns:
            List of similar tools with aggregated scores
        """
        if not query_embeddings:
            return []
        
        # Store all results with their scores
        tool_scores = {}  # tool_name -> list of (score, weight) tuples
        tool_payloads = {}  # tool_name -> payload
        
        # Search with each query embedding
        for query_name, embedding in query_embeddings.items():
            weight = weights.get(query_name, 0.1)  # Default weight if not specified
            
            # Perform search
            results = await self.vector_store.search_similar_tools(
                query_embedding=embedding,
                limit=limit * 3,  # Get more candidates for aggregation
                score_threshold=score_threshold * 0.7 if score_threshold is not None and score_threshold > 0 else score_threshold,  # Lower threshold for individual queries
                filter_dict=filter_dict
            )
            
            # Aggregate scores
            for result in results:
                tool_name = result["tool_name"]
                score = result["score"]
                
                if tool_name not in tool_scores:
                    tool_scores[tool_name] = []
                    tool_payloads[tool_name] = result
                
                tool_scores[tool_name].append((score, weight))
        
        # Calculate weighted average scores
        final_results = []
        for tool_name, score_weights in tool_scores.items():
            # Weighted average of scores
            weighted_score = sum(score * weight for score, weight in score_weights)
            # Normalize by the sum of weights that contributed to this tool
            total_weight = sum(weight for _, weight in score_weights)
            if total_weight > 0:
                normalized_score = weighted_score / total_weight
            else:
                normalized_score = weighted_score
            
            # Add coverage bonus (tools that match more queries get a slight bonus)
            coverage_bonus = len(score_weights) / len(query_embeddings) * 0.1
            final_score = normalized_score * (1 + coverage_bonus)
            
            # Apply threshold if specified
            if score_threshold and final_score < score_threshold:
                continue
            
            result = tool_payloads[tool_name].copy()
            result["score"] = final_score
            result["matched_queries"] = len(score_weights)
            final_results.append(result)
        
        # Sort by score and return top N
        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[:limit]
    
    async def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate tools with scores
            top_k: Number of top results to return
            
        Returns:
            Reranked list of candidates
        """
        if not self.enable_cross_encoder or not self.cross_encoder:
            logger.warning("Cross-encoder not available, returning original candidates")
            return candidates[:top_k]
        
        # Extract original scores
        original_scores = [c.get("score", 0.5) for c in candidates]
        
        # Rerank with cross-encoder
        reranked = await self.cross_encoder.rerank(
            query=query,
            candidates=candidates,
            original_scores=original_scores,
            top_k=top_k,
            score_combination="weighted"
        )
        
        logger.info(f"Cross-encoder reranked {len(candidates)} candidates to top-{len(reranked)}")
        return reranked
    
    async def hybrid_search_with_cross_encoder_reranking(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        available_tools: List[Any] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        rerank_top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Three-stage pipeline: Hybrid search followed by cross-encoder reranking.
        
        This implements the optimal retrieval pipeline:
        1. Stage 1: Hybrid search (semantic + BM25) to get top candidates
        2. Stage 2: Cross-encoder reranking of top candidates
        3. Stage 3: Final filtering and limiting
        
        Args:
            query: Text query
            messages: Conversation messages
            available_tools: List of available tools
            query_embedding: Pre-computed query embedding (optional)
            limit: Final number of results to return
            score_threshold: Minimum score threshold (applied after reranking)
            rerank_top_k: Number of candidates to rerank (default: 3x limit)
            
        Returns:
            List of tools ranked by final pipeline score
        """
        # Extract query text if not provided
        if query is None and messages:
            query = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if query is None:
            raise ValueError("Must provide either query or messages")
        # Determine how many candidates to get for reranking
        if rerank_top_k is None:
            rerank_top_k = min(
                getattr(settings, 'cross_encoder_top_k', 30),
                len(available_tools) if available_tools else 30
            )
        
        # Stage 1: Get more candidates from hybrid search for reranking
        logger.info(f"Stage 1: Hybrid search for top-{rerank_top_k} candidates")
        hybrid_candidates = await self.hybrid_search(
            query=query,
            messages=messages,
            available_tools=available_tools,
            query_embedding=query_embedding,
            limit=rerank_top_k,
            score_threshold=0.0  # No threshold yet, we'll apply after reranking
        )
        
        if not hybrid_candidates:
            return []
        
        # Stage 2: Cross-encoder reranking
        logger.info(f"Stage 2: Cross-encoder reranking {len(hybrid_candidates)} candidates")
        if self.enable_cross_encoder and self.cross_encoder:
            reranked_results = await self.cross_encoder_rerank(
                query=query,
                candidates=hybrid_candidates,
                top_k=min(limit * 2, len(hybrid_candidates))  # Keep more for final filtering
            )
        else:
            logger.warning("Cross-encoder not available, skipping reranking")
            reranked_results = hybrid_candidates
        
        # Stage 3: Apply final threshold and limit
        threshold = score_threshold if score_threshold is not None else self.vector_store.similarity_threshold
        final_results = [r for r in reranked_results if r['score'] >= threshold][:limit]
        
        logger.info(
            f"Pipeline complete: {len(available_tools) if available_tools else 'all'} tools -> "
            f"{len(hybrid_candidates)} hybrid -> {len(reranked_results)} reranked -> "
            f"{len(final_results)} final"
        )
        
        return final_results
    
    async def search(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        available_tools: List[Any] = None,
        strategy: SearchStrategy = SearchStrategy.HYBRID_CROSS_ENCODER,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        query_embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main search interface with configurable strategy.
        
        Args:
            query: Text query
            messages: Conversation messages
            available_tools: List of available tools
            strategy: Search strategy to use
            limit: Maximum number of results
            score_threshold: Minimum score threshold
            query_embedding: Pre-computed query embedding (optional)
            
        Returns:
            List of ranked tools
        """
        logger.info(f"Searching with strategy: {strategy.value}")
        
        if strategy == SearchStrategy.SEMANTIC:
            return await self.semantic_search(
                query=query,
                messages=messages,
                query_embedding=query_embedding,
                available_tools=available_tools,
                limit=limit,
                score_threshold=score_threshold
            )
        
        elif strategy == SearchStrategy.HYBRID:
            return await self.hybrid_search(
                query=query,
                messages=messages,
                available_tools=available_tools,
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
        
        elif strategy == SearchStrategy.CROSS_ENCODER:
            # Extract query if not provided
            if query is None and messages:
                query = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
            if query is None:
                raise ValueError("Must provide either query or messages for cross-encoder")
            
            # First get candidates with semantic search, then rerank
            candidates = await self.semantic_search(
                query=query,
                messages=messages,
                query_embedding=query_embedding,
                available_tools=available_tools,
                limit=min(30, len(available_tools) if available_tools else 30),
                score_threshold=0.0
            )
            return await self.cross_encoder_rerank(
                query=query,
                candidates=candidates,
                top_k=limit
            )
        
        elif strategy == SearchStrategy.HYBRID_CROSS_ENCODER:
            return await self.hybrid_search_with_cross_encoder_reranking(
                query=query,
                messages=messages,
                available_tools=available_tools,
                query_embedding=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
        
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def _extract_tool_names(self, tools: List[Any]) -> List[str]:
        """Extract tool names from various tool formats."""
        tool_names = []
        for tool in tools:
            if hasattr(tool, 'function'):
                tool_names.append(tool.function.name)
            elif isinstance(tool, dict) and 'function' in tool:
                tool_names.append(tool['function']['name'])
            elif isinstance(tool, dict) and 'tool_name' in tool:
                tool_names.append(tool['tool_name'])
            elif isinstance(tool, dict) and 'name' in tool:
                tool_names.append(tool['name'])
            else:
                logger.warning(f"Could not extract name from tool: {tool}")
        return tool_names
    
    def get_available_strategies(self) -> List[SearchStrategy]:
        """Get list of available search strategies based on configuration."""
        strategies = [SearchStrategy.SEMANTIC]
        
        if self.enable_bm25:
            strategies.append(SearchStrategy.HYBRID)
        
        if self.enable_cross_encoder:
            strategies.append(SearchStrategy.CROSS_ENCODER)
            
            if self.enable_bm25:
                strategies.append(SearchStrategy.HYBRID_CROSS_ENCODER)
        
        return strategies
    
    def get_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        stats = {
            "available_strategies": [s.value for s in self.get_available_strategies()],
            "bm25_enabled": self.enable_bm25,
            "cross_encoder_enabled": self.enable_cross_encoder,
        }
        
        if self.cross_encoder:
            stats["cross_encoder_cache"] = self.cross_encoder.get_cache_stats()
        
        return stats