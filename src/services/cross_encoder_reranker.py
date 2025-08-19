"""Cross-encoder reranking service for improved tool ranking accuracy."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sentence_transformers import CrossEncoder

from src.core.config import get_settings
from src.core.models import Tool

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """
    Cross-encoder reranking service that jointly processes query-tool pairs
    for more accurate relevance assessment.
    
    Uses MS-MARCO model fine-tuned for passage ranking tasks.
    """
    
    def __init__(
        self,
        model_name: str = None,
        max_length: int = 512,
        device: str = None,
        cache_size: int = 1000
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM-L-6-v2)
            max_length: Maximum sequence length for model
            device: Device to run model on (cpu/cuda/mps)
            cache_size: Size of LRU cache for predictions
        """
        self.model_name = model_name or getattr(
            settings, 
            'cross_encoder_model', 
            'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self.max_length = max_length
        self.device = device
        
        # Initialize model
        logger.info(f"Loading cross-encoder model: {self.model_name}")
        self.model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device
        )
        
        # Thread pool for CPU-bound model inference
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # Cache for predictions
        self.cache_size = cache_size
        self._cache = {}
        
        # Batch size for inference
        self.batch_size = getattr(settings, 'cross_encoder_batch_size', 32)
        
        logger.info(f"Cross-encoder initialized with batch_size={self.batch_size}")
    
    def _tool_to_text(self, tool: Any) -> str:
        """
        Convert tool to text representation for cross-encoder.
        
        Args:
            tool: Tool object or dictionary
            
        Returns:
            Text representation of tool
        """
        if isinstance(tool, dict):
            # Handle dict format
            if "function" in tool:
                func = tool["function"]
                name = func.get("name", "")
                description = func.get("description", "")
                
                # Include parameter info for richer context
                params = func.get("parameters", {})
                param_props = params.get("properties", {})
                param_names = list(param_props.keys()) if param_props else []
                param_text = f" Parameters: {', '.join(param_names)}" if param_names else ""
                
                return f"{name}: {description}{param_text}"
            else:
                # Simple dict format
                name = tool.get("tool_name", tool.get("name", ""))
                description = tool.get("description", "")
                return f"{name}: {description}"
        elif hasattr(tool, "function"):
            # Tool object
            func = tool.function
            name = func.name if hasattr(func, 'name') else ""
            description = func.description if hasattr(func, 'description') else ""
            
            # Include parameters
            if hasattr(func, 'parameters'):
                params = func.parameters
                param_props = params.get("properties", {}) if isinstance(params, dict) else {}
                param_names = list(param_props.keys()) if param_props else []
                param_text = f" Parameters: {', '.join(param_names)}" if param_names else ""
            else:
                param_text = ""
            
            return f"{name}: {description}{param_text}"
        else:
            # Fallback
            return str(tool)
    
    def _create_pairs(
        self, 
        query: str, 
        candidates: List[Any]
    ) -> List[Tuple[str, str]]:
        """
        Create query-tool pairs for cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate tools
            
        Returns:
            List of (query, tool_text) pairs
        """
        pairs = []
        for candidate in candidates:
            tool_text = self._tool_to_text(candidate)
            pairs.append((query, tool_text))
        return pairs
    
    def _get_cache_key(self, query: str, tool_text: str) -> str:
        """Generate cache key for query-tool pair."""
        return f"{hash(query)}:{hash(tool_text)}"
    
    def _predict_batch(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """
        Run cross-encoder predictions on batch.
        
        Args:
            pairs: List of (query, tool) text pairs
            
        Returns:
            Array of relevance scores
        """
        # Check cache first and track which pairs need computation
        scores = []
        uncached_pairs = []
        uncached_indices = []
        
        for i, (query, tool_text) in enumerate(pairs):
            cache_key = self._get_cache_key(query, tool_text)
            if cache_key in self._cache:
                # Found in cache - use cached score
                scores.append(self._cache[cache_key])
            else:
                # Not in cache - need to compute
                scores.append(None)
                # Only add to uncached_pairs if we haven't seen this exact pair yet
                if (query, tool_text) not in uncached_pairs:
                    uncached_pairs.append((query, tool_text))
                uncached_indices.append(i)
        
        # Predict uncached pairs
        if uncached_pairs:
            logger.debug(f"Computing cross-encoder scores for {len(uncached_pairs)} unique pairs")
            uncached_scores = self.model.predict(
                uncached_pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Create a mapping from pair to score for easy lookup
            pair_to_score = {}
            for (query, tool_text), score in zip(uncached_pairs, uncached_scores):
                cache_key = self._get_cache_key(query, tool_text)
                
                # Maintain cache size limit (simple FIFO)
                if len(self._cache) >= self.cache_size:
                    # Remove oldest entry
                    first_key = next(iter(self._cache))
                    del self._cache[first_key]
                
                self._cache[cache_key] = float(score)
                pair_to_score[(query, tool_text)] = float(score)
            
            # Now fill in the scores for all uncached indices
            for idx in uncached_indices:
                query, tool_text = pairs[idx]
                scores[idx] = pair_to_score[(query, tool_text)]
        
        return np.array(scores)
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None,
        top_k: int = 10,
        score_combination: str = "weighted"
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Search query
            candidates: List of candidate tools with metadata
            original_scores: Original retrieval scores
            top_k: Number of top results to return
            score_combination: How to combine scores ("weighted", "ce_only", "multiplicative")
            
        Returns:
            Reranked list of candidates with updated scores
        """
        if not candidates:
            return []
        
        # Extract tools from candidates
        if isinstance(candidates[0], dict) and "tool" in candidates[0]:
            # Candidates include tool objects - extract them
            tools = [c["tool"] for c in candidates]
            # Keep the original candidates to preserve metadata
            base_candidates = candidates
        else:
            # Candidates are tools themselves
            tools = candidates
            base_candidates = candidates
        
        # Create query-tool pairs
        pairs = self._create_pairs(query, tools)
        
        # Get cross-encoder scores (run in thread to avoid blocking)
        ce_scores = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._predict_batch,
            pairs
        )
        
        # Normalize cross-encoder scores to [0, 1] using sigmoid
        # MS-MARCO models output logits
        ce_scores_normalized = 1 / (1 + np.exp(-ce_scores))
        
        # Combine scores
        combined_results = []
        for i, (candidate, ce_score) in enumerate(zip(base_candidates, ce_scores_normalized)):
            # Flatten the result - if tool is nested, extract its contents
            if isinstance(candidate, dict) and "tool" in candidate:
                # Extract tool contents to top level
                tool_data = candidate["tool"]
                result = tool_data.copy() if isinstance(tool_data, dict) else {"tool": tool_data}
                # Add any other metadata from candidate (but not the nested tool)
                for key, value in candidate.items():
                    if key != "tool" and key not in result:
                        result[key] = value
            else:
                # Already flat structure
                result = candidate.copy()
            
            result["cross_encoder_score"] = float(ce_score)
            
            # Get original score
            if original_scores and i < len(original_scores):
                orig_score = original_scores[i]
            elif "score" in result:
                orig_score = result["score"]
            else:
                orig_score = 0.5  # Default if no score
            
            # Combine scores based on strategy
            if score_combination == "ce_only":
                final_score = ce_score
            elif score_combination == "multiplicative":
                final_score = orig_score * ce_score
            else:  # weighted (default)
                # Weighted combination with emphasis on cross-encoder
                ce_weight = getattr(settings, 'cross_encoder_weight', 0.6)
                orig_weight = 1 - ce_weight
                final_score = (orig_weight * orig_score) + (ce_weight * ce_score)
            
            result["original_score"] = orig_score
            result["score"] = float(final_score)
            combined_results.append(result)
        
        # Sort by final score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        top_results = combined_results[:top_k]
        
        # Log reranking impact
        if original_scores:
            orig_order = [c.get("tool_name", str(i)) for i, c in enumerate(base_candidates)]
            new_order = [r.get("tool_name", str(i)) for i, r in enumerate(top_results)]
            if orig_order[:top_k] != new_order:
                logger.debug(f"Cross-encoder changed ranking: {orig_order[:top_k]} -> {new_order}")
        
        logger.info(
            f"Cross-encoder reranked {len(candidates)} candidates to top-{len(top_results)}, "
            f"cache hit rate: {len([1 for p in pairs if self._get_cache_key(*p) in self._cache])/len(pairs)*100:.1f}%"
        )
        
        return top_results
    
    async def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates that already contain metadata and scores.
        
        Args:
            query: Search query
            candidates: Candidates with tool_name, score, and other metadata
            top_k: Number of results to return
            
        Returns:
            Reranked candidates with cross-encoder scores
        """
        # Extract original scores
        original_scores = [c.get("score", 0.5) for c in candidates]
        
        # Rerank
        reranked = await self.rerank(
            query=query,
            candidates=candidates,
            original_scores=original_scores,
            top_k=top_k
        )
        
        return reranked
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._cache.clear()
        logger.info("Cross-encoder cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_size": len(self._cache),
            "cache_limit": self.cache_size,
            "cache_usage_percent": len(self._cache) / self.cache_size * 100
        }