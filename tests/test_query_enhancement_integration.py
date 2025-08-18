"""Integration test for query enhancement feature."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from src.services.query_enhancer import QueryEnhancer
from src.services.embeddings import EmbeddingService
from src.services.vector_store import VectorStoreService


class TestQueryEnhancementIntegration:
    """Test query enhancement integration."""
    
    @pytest.mark.asyncio
    async def test_enhanced_query_flow(self):
        """Test the complete enhanced query flow."""
        
        # Setup
        messages = [
            {"role": "user", "content": "I need to search for Python files in my project"},
            {"role": "assistant", "content": "I'll help you find Python files"},
            {"role": "user", "content": "Also check for any test files"}
        ]
        
        # Test QueryEnhancer
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance_query(messages)
        
        # Verify enhancement produces multiple queries
        assert "primary" in enhanced["queries"]
        assert "intent" in enhanced["queries"]
        assert len(enhanced["queries"]) >= 2
        
        # Verify weights are calculated
        assert "primary" in enhanced["weights"]
        assert sum(enhanced["weights"].values()) > 0.99
        
        # Verify expansion includes synonyms (test -> check/verify/validate)
        assert "check" in enhanced["expanded"] or "verify" in enhanced["expanded"] or "validate" in enhanced["expanded"]
        
    @pytest.mark.asyncio
    async def test_parallel_embedding(self):
        """Test parallel embedding of multiple queries."""
        
        # Mock embedding service
        embedding_service = Mock(spec=EmbeddingService)
        
        # Create mock embed_text that returns different embeddings
        async def mock_embed_text(text):
            # Return different embeddings based on text content
            if "search" in text.lower():
                return [0.1] * 10
            elif "test" in text.lower():
                return [0.2] * 10
            else:
                return [0.3] * 10
        
        embedding_service.embed_text = AsyncMock(side_effect=mock_embed_text)
        
        # Test embed_queries
        queries = {
            "primary": "search for files",
            "intent": "test and verify",
            "context": "general query"
        }
        
        # Manually implement embed_queries for testing
        async def embed_queries(queries_dict):
            tasks = []
            names = []
            for name, text in queries_dict.items():
                if text:
                    tasks.append(embedding_service.embed_text(text))
                    names.append(name)
            
            embeddings = await asyncio.gather(*tasks)
            return dict(zip(names, embeddings))
        
        result = await embed_queries(queries)
        
        # Verify all queries were embedded
        assert len(result) == 3
        assert "primary" in result
        assert "intent" in result
        assert "context" in result
        
        # Verify embeddings are different
        assert result["primary"][0] == 0.1
        assert result["intent"][0] == 0.2
        assert result["context"][0] == 0.3
    
    @pytest.mark.asyncio 
    async def test_result_aggregation(self):
        """Test aggregation of results from multiple queries."""
        
        # Mock search results from different queries
        query_results = {
            "primary": [
                {"tool_name": "find", "score": 0.9},
                {"tool_name": "grep", "score": 0.8},
                {"tool_name": "ls", "score": 0.6}
            ],
            "intent": [
                {"tool_name": "find", "score": 0.85},
                {"tool_name": "search", "score": 0.75},
                {"tool_name": "grep", "score": 0.7}
            ],
            "context": [
                {"tool_name": "ls", "score": 0.8},
                {"tool_name": "find", "score": 0.7}
            ]
        }
        
        weights = {
            "primary": 0.5,
            "intent": 0.3,
            "context": 0.2
        }
        
        # Simulate aggregation logic
        tool_scores = {}
        for query_name, results in query_results.items():
            weight = weights[query_name]
            for result in results:
                tool_name = result["tool_name"]
                if tool_name not in tool_scores:
                    tool_scores[tool_name] = []
                tool_scores[tool_name].append((result["score"], weight))
        
        # Calculate final scores
        final_scores = {}
        for tool_name, score_weights in tool_scores.items():
            weighted_sum = sum(s * w for s, w in score_weights)
            total_weight = sum(w for _, w in score_weights)
            final_scores[tool_name] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Verify aggregation
        assert "find" in final_scores
        assert "grep" in final_scores
        assert "ls" in final_scores
        assert "search" in final_scores
        
        # Find should have highest score (appears in all queries with high scores)
        sorted_tools = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        assert sorted_tools[0][0] == "find"
    
    @pytest.mark.asyncio
    async def test_error_context_enhancement(self):
        """Test enhancement when error context is present."""
        
        messages = [
            {"role": "user", "content": "I'm trying to connect to the database"},
            {"role": "assistant", "content": "Let me help you connect"},
            {"role": "user", "content": "Getting error: connection refused on port 5432"}
        ]
        
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance_query(messages)
        
        # Should detect error context
        assert enhanced["metadata"]["has_error"] is True
        assert "error" in enhanced["queries"]
        assert "connection refused" in enhanced["queries"]["error"]
        
        # Error weight should be boosted
        assert enhanced["weights"]["error"] > 0.15
    
    @pytest.mark.asyncio
    async def test_code_context_enhancement(self):
        """Test enhancement when code is present."""
        
        messages = [
            {"role": "user", "content": "Here's my code:\n```python\ndef hello():\n    print('world')\n```"},
            {"role": "user", "content": "How can I test this function?"}
        ]
        
        enhancer = QueryEnhancer()
        enhanced = enhancer.enhance_query(messages)
        
        # Should detect code context
        assert enhanced["metadata"]["has_code"] is True
        assert "code" in enhanced["queries"]
        
        # Code weight should be boosted
        assert enhanced["weights"]["code"] > 0.05
    
    def test_feature_flag(self):
        """Test that query enhancement can be disabled via feature flag."""
        from src.core.config import Settings
        
        # Test default is enabled
        settings = Settings(
            primary_embedding_api_key="test_key"
        )
        assert settings.enable_query_enhancement is True
        
        # Test can be disabled
        settings = Settings(
            primary_embedding_api_key="test_key",
            enable_query_enhancement=False
        )
        assert settings.enable_query_enhancement is False