"""Tests for BM25 hybrid search functionality."""

import pytest
from typing import List
import numpy as np

from src.services.bm25_ranker import BM25Ranker, HybridScorer
from src.core.models import Tool, ToolFunction


@pytest.fixture
def sample_tools() -> List[Tool]:
    """Create sample tools for testing."""
    return [
        Tool(
            type="function",
            function=ToolFunction(
                name="file_search_grep",
                description="Search for patterns in files using grep",
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"},
                        "path": {"type": "string", "description": "File or directory path"}
                    },
                    "required": ["pattern", "path"]
                }
            )
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="git_commit",
                description="Create a git commit with a message",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "files": {"type": "array", "description": "Files to commit"}
                    },
                    "required": ["message"]
                }
            )
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="text_finder",
                description="Find text in documents",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Text to find"}
                    },
                    "required": ["query"]
                }
            )
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="database_query",
                description="Execute SQL queries on the database",
                parameters={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL query"},
                        "database": {"type": "string", "description": "Database name"}
                    },
                    "required": ["sql"]
                }
            )
        ),
        Tool(
            type="function",
            function=ToolFunction(
                name="api_request",
                description="Make HTTP API requests",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "API endpoint"},
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]}
                    },
                    "required": ["url", "method"]
                }
            )
        )
    ]


class TestBM25Ranker:
    """Test BM25 ranking functionality."""
    
    def test_initialization(self):
        """Test BM25 ranker initialization."""
        ranker = BM25Ranker()
        assert ranker.variant == "okapi"
        assert ranker.k1 == 1.5
        assert ranker.b == 0.75
        
        # Test custom parameters
        ranker_plus = BM25Ranker(variant="plus", k1=2.0, b=0.5)
        assert ranker_plus.variant == "plus"
        assert ranker_plus.k1 == 2.0
        assert ranker_plus.b == 0.5
    
    def test_text_preprocessing(self):
        """Test text preprocessing."""
        ranker = BM25Ranker()
        
        # Test basic tokenization
        tokens = ranker._preprocess_text("Search files using grep")
        assert "search" in tokens
        assert "files" in tokens
        assert "grep" in tokens
        assert "using" not in tokens  # Stopword removed
        
        # Test technical terms
        tokens = ranker._preprocess_text("file_search_grep")
        assert len(tokens) > 0
        
        # Test empty input
        tokens = ranker._preprocess_text("")
        assert tokens == []
    
    def test_score_tools(self, sample_tools):
        """Test BM25 scoring of tools."""
        ranker = BM25Ranker()
        
        # Test exact match query
        scores = ranker.score_tools("grep search files", sample_tools)
        assert "file_search_grep" in scores
        assert scores["file_search_grep"] > scores.get("git_commit", 0)
        assert scores["file_search_grep"] > scores.get("database_query", 0)
        
        # Test partial match
        scores = ranker.score_tools("find text", sample_tools)
        assert "text_finder" in scores
        assert scores["text_finder"] > 0
        
        # Test no match
        scores = ranker.score_tools("xyz123abc", sample_tools)
        assert all(score == 0 or score < 0.1 for score in scores.values())
    
    def test_top_k_ranking(self, sample_tools):
        """Test top-k retrieval."""
        ranker = BM25Ranker()
        
        scores = ranker.score_tools("search grep files", sample_tools)
        top_2 = ranker.get_top_k(scores, k=2)
        
        assert len(top_2) == 2
        assert top_2[0][1] >= top_2[1][1]  # Sorted by score
        assert "file_search_grep" in [tool for tool, _ in top_2]
    
    def test_batch_scoring(self, sample_tools):
        """Test batch scoring of multiple queries."""
        ranker = BM25Ranker()
        
        queries = ["grep search", "git commit", "database SQL"]
        results = ranker.get_batch_top_k(queries, sample_tools, k=2)
        
        assert len(results) == 3
        assert "file_search_grep" in [tool for tool, _ in results[0]]
        assert "git_commit" in [tool for tool, _ in results[1]]
        assert "database_query" in [tool for tool, _ in results[2]]
    
    def test_bm25_variants(self, sample_tools):
        """Test different BM25 variants."""
        query = "search files"
        
        # Test Okapi BM25
        ranker_okapi = BM25Ranker(variant="okapi")
        scores_okapi = ranker_okapi.score_tools(query, sample_tools)
        
        # Test BM25+
        ranker_plus = BM25Ranker(variant="plus")
        scores_plus = ranker_plus.score_tools(query, sample_tools)
        
        # Test BM25L
        ranker_l = BM25Ranker(variant="l")
        scores_l = ranker_l.score_tools(query, sample_tools)
        
        # All should identify the same top tool
        top_okapi = max(scores_okapi.items(), key=lambda x: x[1])[0]
        top_plus = max(scores_plus.items(), key=lambda x: x[1])[0]
        top_l = max(scores_l.items(), key=lambda x: x[1])[0]
        
        assert top_okapi == top_plus == top_l == "file_search_grep"


class TestHybridScorer:
    """Test hybrid scoring functionality."""
    
    @pytest.fixture
    def semantic_results(self) -> List[dict]:
        """Mock semantic search results."""
        return [
            {"tool_name": "file_search_grep", "score": 0.85},
            {"tool_name": "text_finder", "score": 0.72},
            {"tool_name": "git_commit", "score": 0.45},
            {"tool_name": "database_query", "score": 0.30},
            {"tool_name": "api_request", "score": 0.25}
        ]
    
    @pytest.fixture
    def bm25_scores(self) -> dict:
        """Mock BM25 scores."""
        return {
            "file_search_grep": 15.2,
            "text_finder": 2.1,
            "git_commit": 0.5,
            "database_query": 0.0,
            "api_request": 0.0
        }
    
    def test_weighted_merging(self, sample_tools, semantic_results, bm25_scores):
        """Test weighted score merging."""
        scorer = HybridScorer(semantic_weight=0.7, bm25_weight=0.3)
        
        merged = scorer.merge_scores_weighted(
            semantic_results,
            bm25_scores,
            sample_tools
        )
        
        # Check structure
        assert len(merged) == len(sample_tools)
        assert all("score" in r for r in merged)
        assert all("semantic_score" in r for r in merged)
        assert all("bm25_score" in r for r in merged)
        assert all("bm25_normalized" in r for r in merged)
        
        # Check sorting
        scores = [r["score"] for r in merged]
        assert scores == sorted(scores, reverse=True)
        
        # Check top result (should be file_search_grep)
        assert merged[0]["tool_name"] == "file_search_grep"
        
        # Check score calculation
        top_result = merged[0]
        expected_score = (0.7 * top_result["semantic_score"] + 
                         0.3 * top_result["bm25_normalized"])
        assert abs(top_result["score"] - expected_score) < 0.001
    
    def test_rrf_merging(self, semantic_results, bm25_scores):
        """Test Reciprocal Rank Fusion merging."""
        scorer = HybridScorer()
        
        merged = scorer.merge_scores_rrf(
            semantic_results,
            bm25_scores,
            k=60
        )
        
        # Check structure
        assert all("score" in r for r in merged)
        assert all("semantic_rank" in r for r in merged)
        assert all("bm25_rank" in r for r in merged)
        
        # Check sorting
        scores = [r["score"] for r in merged]
        assert scores == sorted(scores, reverse=True)
        
        # Top result should still be file_search_grep
        assert merged[0]["tool_name"] == "file_search_grep"
        assert merged[0]["semantic_rank"] == 1
        assert merged[0]["bm25_rank"] == 1
    
    def test_normalization_edge_cases(self, sample_tools):
        """Test score normalization edge cases."""
        scorer = HybridScorer()
        
        # Test with all same BM25 scores
        uniform_bm25 = {tool.name: 5.0 for tool in sample_tools}
        semantic_results = [
            {"tool_name": tool.name, "score": 0.5}
            for tool in sample_tools
        ]
        
        merged = scorer.merge_scores_weighted(
            semantic_results,
            uniform_bm25,
            sample_tools
        )
        
        # All normalized BM25 scores should be 0.5
        assert all(r["bm25_normalized"] == 0.5 for r in merged)
        
        # Test with empty BM25 scores
        empty_bm25 = {}
        merged = scorer.merge_scores_weighted(
            semantic_results,
            empty_bm25,
            sample_tools
        )
        
        # Should still work, with 0 BM25 contribution
        assert all(r["bm25_score"] == 0.0 for r in merged)
    
    def test_weight_configurations(self, sample_tools, semantic_results, bm25_scores):
        """Test different weight configurations."""
        # Semantic-only (weight = 1.0)
        scorer_semantic = HybridScorer(semantic_weight=1.0, bm25_weight=0.0)
        merged_semantic = scorer_semantic.merge_scores_weighted(
            semantic_results, bm25_scores, sample_tools
        )
        
        # BM25-only (weight = 1.0)
        scorer_bm25 = HybridScorer(semantic_weight=0.0, bm25_weight=1.0)
        merged_bm25 = scorer_bm25.merge_scores_weighted(
            semantic_results, bm25_scores, sample_tools
        )
        
        # Balanced (weight = 0.5 each)
        scorer_balanced = HybridScorer(semantic_weight=0.5, bm25_weight=0.5)
        merged_balanced = scorer_balanced.merge_scores_weighted(
            semantic_results, bm25_scores, sample_tools
        )
        
        # Different configurations should produce different rankings
        top_semantic = merged_semantic[0]["tool_name"]
        top_bm25 = merged_bm25[0]["tool_name"]
        top_balanced = merged_balanced[0]["tool_name"]
        
        # In this case, all should agree on file_search_grep as top
        assert top_semantic == top_bm25 == top_balanced == "file_search_grep"
        
        # But scores should differ
        assert merged_semantic[0]["score"] != merged_bm25[0]["score"]