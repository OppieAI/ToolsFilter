"""Search service that orchestrates different search strategies."""

import logging
from typing import List, Dict, Any, Optional
from enum import Enum

from src.core.config import get_settings
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.bm25_ranker import BM25Ranker, HybridScorer
from src.services.cross_encoder_reranker import CrossEncoderReranker
from src.services.ltr_service import LTRService
from src.core.models import Tool

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchStrategy(Enum):
    """Available search strategies."""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    CROSS_ENCODER = "cross_encoder"
    HYBRID_CROSS_ENCODER = "hybrid_cross_encoder"
    LTR = "ltr"
    HYBRID_LTR = "hybrid_ltr"


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
        ltr_service: Optional[LTRService] = None,
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
            ltr_service: Optional LTR service instance (created from config if None)
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

        # LTR - use injected or create from config
        self.enable_ltr = getattr(self.config, 'enable_ltr', False)
        self.ltr_service = ltr_service

        if self.enable_ltr and self.ltr_service is None:
            self.ltr_service = self._create_ltr_service()

        if self.enable_ltr and self.ltr_service:
            logger.info("LTR ranking enabled")

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

    def _create_ltr_service(self) -> Optional[LTRService]:
        """Factory method to create LTR service from config."""
        try:
            return LTRService(
                model_path=getattr(self.config, 'ltr_model_path', None),
                bm25_ranker=self.bm25_ranker,
                cross_encoder=self.cross_encoder,
                auto_load=True
            )
        except Exception as e:
            logger.warning(f"Failed to initialize LTR service: {e}")
            self.enable_ltr = False
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
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif isinstance(tool, dict) and 'name' in tool:
                tool_names.append(tool['name'])
            else:
                logger.warning(f"Tool without function name: {tool}")

        # Step 1: Semantic search via Qdrant
        # Get all available tools to ensure complete ranking
        # Use a high limit to ensure we get all tools

        # Debug: Log what we're searching for
        logger.info(f"Hybrid search: Searching for {len(tool_names)} - sample tools: {tool_names[:10]}")

        # IMPORTANT: Request ALL available tools for proper scoring
        # We need to score all tools, not just top N
        search_limit = len(available_tools) if available_tools else 1000

        semantic_results = await self.vector_store.search_similar_tools(
            query_embedding=query_embedding,
            filter_dict={"name": tool_names} if tool_names else None,
            limit=search_limit,  # Get ALL available tools for complete scoring
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
            # For large available_tools, we should consider more candidates
            # But cap it to avoid excessive computation
            default_top_k = getattr(settings, 'cross_encoder_top_k', 30)
            if available_tools:
                # Scale up based on available tools, but cap at 100 for performance
                rerank_top_k = min(
                    max(default_top_k, len(available_tools) // 5),  # At least 20% of tools
                    100  # Cap at 100 for performance
                )
            else:
                rerank_top_k = default_top_k

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

    async def multi_criteria_search(
        self,
        query: str,
        available_tools: List[Any],
        query_embedding: Optional[List[float]] = None,
        semantic_limit: int = 100,
        exact_match_boost: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Two-phase retrieval: Cast wide net with multiple search criteria.

        Args:
            query: Search query
            available_tools: List of available tools to search within
            query_embedding: Pre-computed query embedding
            semantic_limit: Limit for semantic search
            exact_match_boost: Whether to boost exact name matches

        Returns:
            Combined, deduplicated candidates with scores
        """
        if not available_tools:
            return []

        # Extract tool names for filtering
        tool_names = self._extract_tool_names(available_tools)
        base_filter = {"name": tool_names} if tool_names else None

        all_candidates = {}  # Use dict for deduplication by tool_name

        # Phase 1: Semantic search (understanding-based)
        logger.debug(f"Phase 1: Semantic search for top {semantic_limit}")
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_text(query)

        semantic_results = await self.vector_store.search_similar_tools(
            query_embedding=query_embedding,
            filter_dict=base_filter,
            limit=semantic_limit,
            score_threshold=0.0  # Get all candidates
        )

        for result in semantic_results:
            tool_name = result['tool_name']
            if tool_name not in all_candidates:
                all_candidates[tool_name] = result
                all_candidates[tool_name]['match_types'] = ['semantic']
                all_candidates[tool_name]['semantic_score'] = result['score']
            else:
                all_candidates[tool_name]['match_types'].append('semantic')
                all_candidates[tool_name]['semantic_score'] = result['score']

        # Phase 2: Exact name matches (user knows what they want)
        query_lower = query.lower()
        query_words = set(query_lower.split())

        logger.debug("Phase 2: Exact name matching")
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchText

        # Check for exact tool names in query
        exact_name_filter = Filter(
            must=[
                FieldCondition(
                    key="name",
                    match=MatchAny(any=tool_names)
                )
            ]
        )

        # Find tools whose names appear in the query
        matching_names = []
        for tool_name in tool_names:
            name_lower = tool_name.lower()
            # Check if tool name appears in query
            if name_lower in query_lower or name_lower.replace('_', ' ') in query_lower:
                matching_names.append(tool_name)

        if matching_names:
            logger.debug(f"Found exact name matches: {matching_names}")
            exact_filter = Filter(
                must=[
                    FieldCondition(
                        key="name",
                        match=MatchAny(any=matching_names)
                    )
                ]
            )

            exact_results = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=exact_filter,
                limit=len(matching_names) * 2  # Get all matches
            )

            if exact_results[0]:
                for point in exact_results[0]:
                    tool_name = point.payload.get("name")
                    if tool_name not in all_candidates:
                        all_candidates[tool_name] = {
                            'tool_name': tool_name,
                            'score': 1.0 if exact_match_boost else 0.9,  # High score for exact match
                            'description': point.payload.get("description"),
                            'category': point.payload.get("category"),
                            'parameters': point.payload.get("parameters"),
                            'original': point.payload.get("original"),
                            'match_types': ['exact_name']
                        }
                    else:
                        all_candidates[tool_name]['match_types'].append('exact_name')
                        if exact_match_boost:
                            all_candidates[tool_name]['score'] = max(
                                all_candidates[tool_name]['score'],
                                0.95
                            )

        # Phase 3: Parameter name matches (user mentions specific params)
        logger.debug("Phase 3: Parameter name matching")
        param_matches = []

        # Look for parameter names in query
        for tool in available_tools:
            tool_name = self._get_tool_name_from_tool(tool)
            if tool_name not in all_candidates:  # Don't re-process
                params = self._get_tool_parameters(tool)
                if params and params.get("properties"):
                    param_names = list(params["properties"].keys())
                    # Check if any parameter name appears in query
                    for param in param_names:
                        if param.lower() in query_lower:
                            param_matches.append(tool_name)
                            break

        if param_matches:
            logger.debug(f"Found parameter matches: {param_matches[:10]}")  # Log first 10
            param_filter = Filter(
                must=[
                    FieldCondition(
                        key="name",
                        match=MatchAny(any=param_matches)
                    )
                ]
            )

            param_results = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=param_filter,
                limit=len(param_matches)
            )

            if param_results[0]:
                for point in param_results[0]:
                    tool_name = point.payload.get("name")
                    if tool_name not in all_candidates:
                        all_candidates[tool_name] = {
                            'tool_name': tool_name,
                            'score': 0.7,  # Decent score for param match
                            'description': point.payload.get("description"),
                            'category': point.payload.get("category"),
                            'parameters': point.payload.get("parameters"),
                            'original': point.payload.get("original"),
                            'match_types': ['param_match']
                        }
                    else:
                        all_candidates[tool_name]['match_types'].append('param_match')

        # Phase 4: Description keyword matches (fallback for missed tools)
        # This could be expensive, so only do for tools not yet found
        missing_tools = [name for name in tool_names if name not in all_candidates]

        if missing_tools and len(missing_tools) < 100:  # Limit to avoid performance issues
            logger.debug(f"Phase 4: Checking {len(missing_tools)} missing tools for description matches")

            # Use searchable_text field if we have it
            keyword_filter = Filter(
                must=[
                    FieldCondition(
                        key="name",
                        match=MatchAny(any=missing_tools)
                    )
                ]
            )

            keyword_results = self.vector_store.client.scroll(
                collection_name=self.vector_store.collection_name,
                scroll_filter=keyword_filter,
                limit=len(missing_tools)
            )

            if keyword_results[0]:
                for point in keyword_results[0]:
                    # Check if description contains query keywords
                    description = point.payload.get("description", "").lower()
                    if any(word in description for word in query_words if len(word) > 3):
                        tool_name = point.payload.get("name")
                        if tool_name not in all_candidates:
                            all_candidates[tool_name] = {
                                'tool_name': tool_name,
                                'score': 0.5,  # Lower score for description match
                                'description': point.payload.get("description"),
                                'category': point.payload.get("category"),
                                'parameters': point.payload.get("parameters"),
                                'original': point.payload.get("original"),
                                'match_types': ['description_match']
                            }

        # Convert to list and sort by score
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.info(
            f"Multi-criteria search found {len(candidates)} candidates: "
            f"{sum(1 for c in candidates if 'semantic' in c.get('match_types', []))} semantic, "
            f"{sum(1 for c in candidates if 'exact_name' in c.get('match_types', []))} exact, "
            f"{sum(1 for c in candidates if 'param_match' in c.get('match_types', []))} param, "
            f"{sum(1 for c in candidates if 'description_match' in c.get('match_types', []))} description"
        )

        return candidates

    def _get_tool_name_from_tool(self, tool: Any) -> str:
        """Extract tool name from tool object."""
        if hasattr(tool, 'name'):
            return tool.name
        elif isinstance(tool, dict):
            return tool.get("name", tool.get("tool_name", ""))
        return ""

    def _get_tool_parameters(self, tool: Any) -> Dict[str, Any]:
        """Extract parameters from tool object."""
        if hasattr(tool, 'parameters'):
            params = tool.parameters
            # Convert ToolParameters object to dict if needed
            if params and hasattr(params, 'model_dump'):
                return params.model_dump()
            elif params and hasattr(params, 'dict'):
                return params.dict()
            else:
                return params if params else {}
        elif isinstance(tool, dict):
            return tool.get("parameters", {})
        return {}

    async def ltr_search(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        available_tools: List[Any] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        candidate_limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search using Learning to Rank model with multi-criteria candidate retrieval.

        Args:
            query: Search query
            messages: Conversation messages
            available_tools: List of available tools
            query_embedding: Pre-computed query embedding (optional)
            limit: Final number of results
            candidate_limit: Number of candidates to get for LTR

        Returns:
            List of tools ranked by LTR model
        """
        if not self.enable_ltr or not self.ltr_service:
            logger.warning("LTR not available, falling back to semantic search")
            return await self.semantic_search(
                query=query, messages=messages, query_embedding=query_embedding,
                available_tools=available_tools, limit=limit
            )

        # Extract query if not provided
        if query is None and messages:
            query = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if query is None:
            raise ValueError("Must provide either query or messages for LTR")

        # Use multi-criteria search to get broader candidate pool
        candidates = await self.multi_criteria_search(
            query=query,
            available_tools=available_tools,
            query_embedding=query_embedding,
            semantic_limit=candidate_limit,
            exact_match_boost=True
        )

        # Apply LTR ranking to the broader candidate set
        ranked = await self.ltr_service.rank_tools(
            query=query,
            candidates=candidates,
            top_k=limit
        )

        return ranked

    async def hybrid_ltr_search(
        self,
        query: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        available_tools: List[Any] = None,
        query_embedding: Optional[List[float]] = None,
        limit: int = 10,
        candidate_limit: int = 150
    ) -> List[Dict[str, Any]]:
        """
        Advanced pipeline: Multi-criteria + BM25 + Cross-encoder + LTR.

        Four-stage pipeline:
        1. Multi-criteria search (semantic + exact + param + description)
        2. BM25 scoring on candidates
        3. Cross-encoder reranking (if enabled)
        4. LTR final ranking

        Args:
            query: Search query
            messages: Conversation messages
            available_tools: List of available tools
            query_embedding: Pre-computed query embedding (optional)
            limit: Final number of results
            candidate_limit: Number of candidates to get for LTR

        Returns:
            List of tools ranked by LTR model
        """
        if not self.enable_ltr or not self.ltr_service:
            logger.warning("LTR not available, falling back to hybrid search")
            return await self.hybrid_search(
                query=query, messages=messages, available_tools=available_tools,
                query_embedding=query_embedding, limit=limit
            )

        # Extract query if not provided
        if query is None and messages:
            query = " ".join(msg["content"] for msg in messages if msg.get("role") == "user")
        if query is None:
            raise ValueError("Must provide either query or messages for LTR")

        # Stage 1: Multi-criteria search for broad candidate pool
        candidates = await self.multi_criteria_search(
            query=query,
            available_tools=available_tools,
            query_embedding=query_embedding,
            semantic_limit=candidate_limit,
            exact_match_boost=True
        )

        # Stage 2: Add BM25 scores if enabled
        if self.enable_bm25 and self.bm25_ranker and candidates:
            # Convert candidates to Tool objects for BM25
            from src.core.models import Tool, ToolParameters
            candidate_tools = []
            for c in candidates:
                # Create Tool object from candidate dict
                params = c.get('parameters')
                if params and isinstance(params, dict) and params:
                    tool_params = ToolParameters(**params)
                else:
                    tool_params = {}
                
                tool_obj = Tool(
                    name=c.get('tool_name', c.get('name', '')),
                    description=c.get('description', ''),
                    parameters=tool_params
                )
                candidate_tools.append(tool_obj)

            # Get BM25 scores
            bm25_scores = self.bm25_ranker.score_tools(query, candidate_tools)

            # Add BM25 scores to candidates
            for c in candidates:
                tool_name = c.get('tool_name', '')
                c['bm25_score'] = bm25_scores.get(tool_name, 0.0)

                # Update combined score if BM25 is significant
                if c['bm25_score'] > 0.5:
                    # Weighted combination
                    semantic = c.get('semantic_score', c.get('score', 0))
                    c['hybrid_score'] = (0.7 * semantic + 0.3 * c['bm25_score'])
                    c['score'] = c['hybrid_score']

        # Stage 3: Optional cross-encoder reranking
        if self.enable_cross_encoder and self.cross_encoder and len(candidates) > 20:
            # Only rerank if we have many candidates
            candidates = await self.cross_encoder_rerank(
                query=query,
                candidates=candidates,
                top_k=min(candidate_limit, len(candidates))
            )

        # Stage 4: LTR final ranking
        ranked = await self.ltr_service.rank_tools(
            query=query,
            candidates=candidates,
            top_k=limit
        )

        logger.info(
            f"Hybrid-LTR pipeline: {len(available_tools) if available_tools else 'all'} tools -> "
            f"{len(candidates)} multi-criteria -> "
            f"{len(ranked)} final LTR ranked"
        )

        return ranked

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
        # Debug logging
        logger.info(f"[DEBUG] Search called with:")
        logger.info(f"  - strategy: {strategy.value}")
        logger.info(f"  - limit: {limit}")
        logger.info(f"  - score_threshold: {score_threshold}")
        logger.info(f"  - available_tools count: {len(available_tools) if available_tools else 0}")
        logger.info(f"  - messages: {len(messages) if messages else 0} messages")
        logger.info(f"  - query: {query[:100] if query else 'None'}...")
        
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

        elif strategy == SearchStrategy.LTR:
            return await self.ltr_search(
                query=query,
                messages=messages,
                available_tools=available_tools,
                query_embedding=query_embedding,
                limit=limit
            )

        elif strategy == SearchStrategy.HYBRID_LTR:
            return await self.hybrid_ltr_search(
                query=query,
                messages=messages,
                available_tools=available_tools,
                query_embedding=query_embedding,
                limit=limit
            )

        else:
            raise ValueError(f"Unknown search strategy: {strategy}")

    def _extract_tool_names(self, tools: List[Any]) -> List[str]:
        """Extract tool names from various tool formats."""
        tool_names = []
        for tool in tools:
            if hasattr(tool, 'name'):
                tool_names.append(tool.name)
            elif isinstance(tool, dict) and 'name' in tool:
                tool_names.append(tool['name'])
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

        if self.enable_ltr:
            strategies.append(SearchStrategy.LTR)

            if self.enable_bm25:
                strategies.append(SearchStrategy.HYBRID_LTR)

        return strategies

    def get_stats(self) -> Dict[str, Any]:
        """Get search service statistics."""
        stats = {
            "available_strategies": [s.value for s in self.get_available_strategies()],
            "bm25_enabled": self.enable_bm25,
            "cross_encoder_enabled": self.enable_cross_encoder,
            "ltr_enabled": self.enable_ltr,
        }

        if self.cross_encoder:
            stats["cross_encoder_cache"] = self.cross_encoder.get_cache_stats()

        if self.ltr_service:
            stats["ltr_stats"] = self.ltr_service.get_stats()

        return stats
