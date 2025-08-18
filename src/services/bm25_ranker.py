"""BM25 ranking service for hybrid search using production-grade rank-bm25 library."""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from rank_bm25 import BM25Okapi, BM25Plus, BM25L

# Import only what we need from NLTK to avoid WordNet dependency
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from src.core.models import Tool

logger = logging.getLogger(__name__)


class BM25Ranker:
    """
    Production-grade BM25 ranking using rank-bm25 library.
    Supports multiple BM25 variants and professional tokenization.
    """

    def __init__(self,
                 variant: str = "okapi",
                 k1: float = 1.5,
                 b: float = 0.75,
                 delta: float = 0.5,
                 use_stemming: bool = False,
                 remove_stopwords: bool = True,
                 lowercase: bool = True):
        """
        Initialize BM25 ranker with production-grade settings.

        Args:
            variant: BM25 variant - "okapi" (standard), "plus", or "l"
            k1: Term frequency saturation (higher = less saturation)
            b: Length normalization (0 = no normalization, 1 = full)
            delta: Delta parameter for BM25Plus (small constant)
            use_stemming: Whether to use Porter stemming
            remove_stopwords: Whether to remove English stopwords
            lowercase: Whether to lowercase text
        """
        self.variant = variant
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase

        # Initialize NLP components
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()

        # Add technical stopwords that aren't useful for tool search
        self.stop_words.update({
            'use', 'using', 'used', 'uses', 'get', 'set', 'make', 'made',
            'function', 'method', 'api', 'tool', 'parameter', 'param',
            'value', 'values', 'return', 'returns', 'data', 'info'
        })

    def _preprocess_text(self, text: str) -> List[str]:
        """
        Production-grade text preprocessing using NLTK.

        Args:
            text: Input text

        Returns:
            List of processed tokens
        """
        # Lowercase if configured
        if self.lowercase:
            text = text.lower()

        # Tokenize using NLTK's word tokenizer (handles punctuation well)
        tokens = word_tokenize(text)

        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip if too short or only punctuation
            if len(token) < 2 or not any(c.isalnum() for c in token):
                continue

            # Skip stopwords if configured
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue

            # Apply stemming if configured
            if self.stemmer:
                token = self.stemmer.stem(token)

            processed_tokens.append(token)

        return processed_tokens

    def _tool_to_searchable_text(self, tool: Tool) -> str:
        """
        Convert tool to optimized searchable text.

        Args:
            tool: Tool object

        Returns:
            Searchable text representation
        """
        parts = []

        # Function name gets highest weight (repeat 3x for emphasis)
        func_name = tool.function.name
        parts.extend([func_name] * 3)

        # Split snake_case/camelCase names for better matching
        import re
        name_parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', func_name)
        name_parts = re.sub(r'[_\-]', ' ', name_parts)
        parts.append(name_parts)

        # Description (medium weight)
        if tool.function.description:
            parts.append(tool.function.description)

        # Parameter names and descriptions (lower weight)
        if tool.function.parameters:
            properties = tool.function.parameters.get("properties", {})
            for param_name, param_info in properties.items():
                parts.append(param_name)
                if isinstance(param_info, dict):
                    if "description" in param_info:
                        parts.append(param_info["description"])
                    # Add enum values if present (useful for matching)
                    if "enum" in param_info:
                        parts.extend(str(v) for v in param_info["enum"])

        return " ".join(parts)

    def score_tools(self, query: str, available_tools: List[Tool]) -> Dict[str, float]:
        """
        Score available tools using BM25 with rank-bm25 library.

        Args:
            query: Search query
            available_tools: List of tools to score

        Returns:
            Dictionary mapping tool names to BM25 scores
        """
        if not available_tools:
            return {}

        # Preprocess query
        query_tokens = self._preprocess_text(query)
        if not query_tokens:
            logger.warning(f"Query '{query}' resulted in no tokens after preprocessing")
            return {tool.function.name: 0.0 for tool in available_tools}

        # Build corpus from available tools
        corpus_texts = []
        tool_names = []

        for tool in available_tools:
            text = self._tool_to_searchable_text(tool)
            corpus_texts.append(text)
            tool_names.append(tool.function.name)

        # Tokenize corpus
        tokenized_corpus = [self._preprocess_text(text) for text in corpus_texts]

        # Choose BM25 variant
        if self.variant == "plus":
            # BM25+ adds a small constant to prevent zero scores
            bm25 = BM25Plus(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        elif self.variant == "l":
            # BM25L uses a different length normalization
            bm25 = BM25L(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        else:
            # Standard BM25 (Okapi)
            bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        # Get scores for all documents
        scores = bm25.get_scores(query_tokens)

        # Map scores to tool names
        tool_scores = {}
        for tool_name, score in zip(tool_names, scores):
            tool_scores[tool_name] = float(score)

        # Log statistics for debugging
        if logger.isEnabledFor(logging.DEBUG):
            non_zero = sum(1 for s in scores if s > 0)
            logger.debug(f"BM25 scoring: {non_zero}/{len(scores)} tools have non-zero scores")
            if scores.any():
                logger.debug(f"Score range: {scores.min():.3f} - {scores.max():.3f}")

        return tool_scores

    def get_top_k(self, scores: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top-k tools by score.

        Args:
            scores: Dictionary of tool names to scores
            k: Number of top tools to return

        Returns:
            List of (tool_name, score) tuples sorted by score
        """
        sorted_tools = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_tools[:k]

    def get_batch_top_k(self, queries: List[str], available_tools: List[Tool], k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Efficiently score multiple queries against the same tool set.

        Args:
            queries: List of search queries
            available_tools: List of tools to score
            k: Number of top tools per query

        Returns:
            List of top-k results for each query
        """
        if not available_tools or not queries:
            return []

        # Build corpus once
        corpus_texts = []
        tool_names = []

        for tool in available_tools:
            text = self._tool_to_searchable_text(tool)
            corpus_texts.append(text)
            tool_names.append(tool.function.name)

        # Tokenize corpus once
        tokenized_corpus = [self._preprocess_text(text) for text in corpus_texts]

        # Create BM25 model once
        if self.variant == "plus":
            bm25 = BM25Plus(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        elif self.variant == "l":
            bm25 = BM25L(tokenized_corpus, k1=self.k1, b=self.b, delta=self.delta)
        else:
            bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

        # Score each query
        results = []
        for query in queries:
            query_tokens = self._preprocess_text(query)
            if not query_tokens:
                results.append([])
                continue

            scores = bm25.get_scores(query_tokens)
            tool_scores = {name: float(score) for name, score in zip(tool_names, scores)}
            top_k = self.get_top_k(tool_scores, k)
            results.append(top_k)

        return results


class HybridScorer:
    """
    Combines BM25 and semantic scores using various strategies.
    """

    def __init__(self,
                 bm25_ranker: Optional[BM25Ranker] = None,
                 semantic_weight: float = 0.7,
                 bm25_weight: float = 0.3):
        """
        Initialize hybrid scorer.

        Args:
            bm25_ranker: BM25 ranker instance (creates default if None)
            semantic_weight: Weight for semantic scores
            bm25_weight: Weight for BM25 scores
        """
        self.bm25_ranker = bm25_ranker or BM25Ranker()
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

    def merge_scores_weighted(self,
                            semantic_results: List[Dict[str, Any]],
                            bm25_scores: Dict[str, float],
                            available_tools: List[Tool]) -> List[Dict[str, Any]]:
        """
        Merge scores using normalized weighted sum.

        Args:
            semantic_results: Results from semantic search
            bm25_scores: BM25 scores for tools
            available_tools: List of available tools

        Returns:
            Merged and sorted results
        """
        # Normalize BM25 scores to 0-1 range
        bm25_values = list(bm25_scores.values())
        bm25_normalized = {}

        if bm25_values:
            min_bm25 = min(bm25_values)
            max_bm25 = max(bm25_values)
            range_bm25 = max_bm25 - min_bm25

            if range_bm25 > 0:
                for tool_name, score in bm25_scores.items():
                    bm25_normalized[tool_name] = (score - min_bm25) / range_bm25
            else:
                # All scores are the same
                bm25_normalized = {k: 0.5 for k in bm25_scores}

        # Create semantic lookup (already 0-1 range)
        semantic_lookup = {
            result['tool_name']: result['score']
            for result in semantic_results
        }

        # Merge scores
        merged_results = []
        for tool in available_tools:
            tool_name = tool.function.name

            # Get individual scores
            semantic_score = semantic_lookup.get(tool_name, 0.0)
            bm25_norm = bm25_normalized.get(tool_name, 0.0)

            # Weighted combination
            final_score = (self.semantic_weight * semantic_score +
                          self.bm25_weight * bm25_norm)

            # Log detailed scoring for debugging
            # logger.debug(f"Tool: {tool_name}")
            # logger.debug(f"  Semantic: {semantic_score:.3f}")
            # logger.debug(f"  BM25 raw: {bm25_scores.get(tool_name, 0.0):.3f}")
            # logger.debug(f"  BM25 normalized: {bm25_norm:.3f}")
            # logger.debug(f"  Final weighted: {final_score:.3f} = {self.semantic_weight}*{semantic_score:.3f} + {self.bm25_weight}*{bm25_norm:.3f}")

            merged_results.append({
                'tool_name': tool_name,
                'score': final_score,
                'semantic_score': semantic_score,
                'bm25_score': bm25_scores.get(tool_name, 0.0),
                'bm25_normalized': bm25_norm,
                'original': tool.model_dump()
            })

        # Sort by final score
        return sorted(merged_results, key=lambda x: x['score'], reverse=True)

    def merge_scores_rrf(self,
                        semantic_results: List[Dict[str, Any]],
                        bm25_scores: Dict[str, float],
                        k: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion - combines rankings without normalization.
        Used by Elasticsearch and other production systems.

        Args:
            semantic_results: Results from semantic search
            bm25_scores: BM25 scores for tools
            k: Constant for RRF formula (typically 60)

        Returns:
            Merged results using RRF
        """
        # Get rankings from semantic results
        semantic_ranks = {}
        for rank, result in enumerate(semantic_results, 1):
            semantic_ranks[result['tool_name']] = rank

        # Get rankings from BM25
        bm25_sorted = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranks = {}
        for rank, (tool_name, _) in enumerate(bm25_sorted, 1):
            bm25_ranks[tool_name] = rank

        # Calculate RRF scores
        all_tools = set(semantic_ranks.keys()) | set(bm25_ranks.keys())
        rrf_results = []

        for tool_name in all_tools:
            # RRF formula: score = Σ(1 / (k + rank))
            score = 0

            # Semantic contribution
            sem_rank = semantic_ranks.get(tool_name, len(semantic_results) + 1)
            score += 1 / (k + sem_rank)

            # BM25 contribution
            bm25_rank = bm25_ranks.get(tool_name, len(bm25_scores) + 1)
            score += 1 / (k + bm25_rank)

            rrf_results.append({
                'tool_name': tool_name,
                'score': score,
                'semantic_rank': sem_rank,
                'bm25_rank': bm25_rank
            })

        return sorted(rrf_results, key=lambda x: x['score'], reverse=True)
