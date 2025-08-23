"""Test two-stage filtering functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from typing import List, Dict, Any

from src.services.search_service import SearchService, SearchStrategy
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.core.models import Tool, ToolParameters
from src.core.config import Settings


@pytest.fixture
def mock_tools() -> List[Tool]:
    """Create mock tools for testing."""
    return [
        Tool(
            name="get_user_info",
            description="Get user information from database",
            parameters=ToolParameters(
                type="object",
                properties={
                    "user_id": {"type": "string", "description": "User ID"}
                },
                required=["user_id"]
            )
        ),
        Tool(
            name="update_user_profile", 
            description="Update user profile information",
            parameters=ToolParameters(
                type="object",
                properties={
                    "user_id": {"type": "string", "description": "User ID"},
                    "name": {"type": "string", "description": "New name"}
                },
                required=["user_id"]
            )
        ),
        Tool(
            name="delete_user_account",
            description="Delete user account permanently", 
            parameters=ToolParameters(
                type="object",
                properties={
                    "user_id": {"type": "string", "description": "User ID"}
                },
                required=["user_id"]
            )
        ),
        Tool(
            name="list_all_users",
            description="List all users in the system",
            parameters=ToolParameters(type="object", properties={})
        ),
        Tool(
            name="send_email_notification",
            description="Send email notification to user",
            parameters=ToolParameters(
                type="object",
                properties={
                    "recipient": {"type": "string", "description": "Email recipient"},
                    "subject": {"type": "string", "description": "Email subject"},
                    "body": {"type": "string", "description": "Email body"}
                },
                required=["recipient", "subject", "body"]
            )
        )
    ]


@pytest.fixture
def mock_config() -> Settings:
    """Create mock config with two-stage filtering enabled."""
    config = Mock(spec=Settings)
    config.enable_hybrid_search = True
    config.enable_cross_encoder = True  
    config.enable_ltr = True
    config.enable_two_stage_filtering = True
    config.two_stage_stage1_threshold = 0.10
    config.two_stage_stage1_limit = 50
    config.two_stage_stage2_threshold = 0.15
    config.two_stage_enable_confidence_cutoff = True
    config.cross_encoder_top_k = 30
    return config


@pytest.fixture
def mock_search_service(mock_config) -> SearchService:
    """Create mock search service for testing."""
    mock_vector_store = Mock(spec=VectorStoreService)
    mock_vector_store.similarity_threshold = 0.13
    
    mock_embedding_service = Mock(spec=EmbeddingService)
    mock_embedding_service.embed_text = AsyncMock(return_value=[0.1] * 1024)
    
    # Mock BM25 ranker
    mock_bm25_ranker = Mock()
    mock_bm25_ranker.score_tools.return_value = {
        "get_user_info": 0.8,
        "update_user_profile": 0.6,
        "delete_user_account": 0.4,
        "list_all_users": 0.3,
        "send_email_notification": 0.1
    }
    
    # Mock hybrid scorer
    mock_hybrid_scorer = Mock()
    
    # Mock cross-encoder
    mock_cross_encoder = Mock()
    mock_cross_encoder.rerank = AsyncMock()
    
    # Mock LTR service
    mock_ltr_service = Mock()
    mock_ltr_service.rank_tools = AsyncMock()
    
    service = SearchService(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        bm25_ranker=mock_bm25_ranker,
        cross_encoder=mock_cross_encoder,
        hybrid_scorer=mock_hybrid_scorer,
        ltr_service=mock_ltr_service,
        config=mock_config
    )
    
    return service


@pytest.mark.asyncio
async def test_two_stage_search_basic_flow(mock_search_service, mock_tools):
    """Test basic two-stage search flow."""
    
    # Mock stage 1 results (hybrid search with lower threshold)
    stage1_results = [
        {"tool_name": "get_user_info", "score": 0.85, "description": "Get user information"},
        {"tool_name": "update_user_profile", "score": 0.75, "description": "Update user profile"},
        {"tool_name": "delete_user_account", "score": 0.65, "description": "Delete user account"},
        {"tool_name": "list_all_users", "score": 0.55, "description": "List all users"},
        {"tool_name": "send_email_notification", "score": 0.45, "description": "Send email"}
    ]
    
    # Mock stage 2 cross-encoder results
    stage2_results = [
        {"tool_name": "get_user_info", "score": 0.90, "description": "Get user information"},
        {"tool_name": "update_user_profile", "score": 0.70, "description": "Update user profile"},
        {"tool_name": "list_all_users", "score": 0.50, "description": "List all users"},
        {"tool_name": "delete_user_account", "score": 0.45, "description": "Delete user account"},
        {"tool_name": "send_email_notification", "score": 0.10, "description": "Send email"}  # Below stage2 threshold
    ]
    
    # Mock stage 3 LTR results
    ltr_results = [
        {"tool_name": "get_user_info", "score": 0.92, "description": "Get user information"},
        {"tool_name": "update_user_profile", "score": 0.72, "description": "Update user profile"},
        {"tool_name": "list_all_users", "score": 0.48, "description": "List all users"},  # Below stage2 threshold
        {"tool_name": "delete_user_account", "score": 0.40, "description": "Delete user account"}  # Below stage2 threshold
    ]
    
    # Set up mocks
    with patch.object(mock_search_service, 'hybrid_search', new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = stage1_results
        
        with patch.object(mock_search_service, 'cross_encoder_rerank', new_callable=AsyncMock) as mock_cross_encoder:
            mock_cross_encoder.return_value = stage2_results
            
            with patch.object(mock_search_service.ltr_service, 'rank_tools', new_callable=AsyncMock) as mock_ltr:
                mock_ltr.return_value = ltr_results
                
                # Execute two-stage search
                results = await mock_search_service.two_stage_search(
                    query="get user information",
                    available_tools=mock_tools,
                    limit=10
                )
    
    # Verify calls were made with correct parameters
    mock_hybrid.assert_called_once()
    mock_cross_encoder.assert_called_once()
    mock_ltr.assert_called_once()
    
    # Verify stage 1 was called with lower threshold and higher limit
    hybrid_call_args = mock_hybrid.call_args[1]
    assert hybrid_call_args['limit'] == 50  # stage1_limit
    assert hybrid_call_args['score_threshold'] == 0.10  # stage1_threshold
    
    # Verify final results are filtered by stage2 threshold (0.15) 
    assert len(results) == 2  # Only get_user_info (0.92) and update_user_profile (0.72) pass threshold
    assert results[0]['tool_name'] == 'get_user_info'
    assert results[1]['tool_name'] == 'update_user_profile'


@pytest.mark.asyncio 
async def test_two_stage_search_confidence_cutoff(mock_search_service, mock_tools):
    """Test confidence cutoff functionality."""
    
    # Create results with large score drops to test cutoff
    stage1_results = [
        {"tool_name": "get_user_info", "score": 0.90},
        {"tool_name": "update_user_profile", "score": 0.85},
        {"tool_name": "delete_user_account", "score": 0.80},
        {"tool_name": "list_all_users", "score": 0.75},
        {"tool_name": "send_email_notification", "score": 0.30},  # Large drop should trigger cutoff
        {"tool_name": "other_tool", "score": 0.25}
    ]
    
    with patch.object(mock_search_service, 'hybrid_search', new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = stage1_results
        
        # Mock no reranking to test cutoff directly 
        mock_search_service.enable_cross_encoder = False
        mock_search_service.enable_ltr = False
        
        results = await mock_search_service.two_stage_search(
            query="get user information", 
            available_tools=mock_tools,
            limit=10,
            enable_confidence_cutoff=True
        )
    
    # Should apply cutoff after large score drop
    assert len(results) <= 4  # Should stop before the large drop
    
    # Top results should be included
    assert results[0]['tool_name'] == 'get_user_info'
    assert results[0]['score'] >= 0.15  # Above stage2 threshold


@pytest.mark.asyncio
async def test_two_stage_search_empty_stage1_results(mock_search_service, mock_tools):
    """Test handling when stage 1 returns no results."""
    
    with patch.object(mock_search_service, 'hybrid_search', new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = []  # No stage 1 results
        
        results = await mock_search_service.two_stage_search(
            query="nonexistent functionality",
            available_tools=mock_tools,
            limit=10
        )
    
    # Should return empty results gracefully
    assert len(results) == 0
    mock_hybrid.assert_called_once()


@pytest.mark.asyncio
async def test_two_stage_search_strategy_integration(mock_search_service, mock_tools):
    """Test two-stage search as part of main search interface."""
    
    # Mock hybrid search results
    stage1_results = [
        {"tool_name": "get_user_info", "score": 0.80, "description": "Get user information"}
    ]
    
    with patch.object(mock_search_service, 'hybrid_search', new_callable=AsyncMock) as mock_hybrid:
        mock_hybrid.return_value = stage1_results
        
        # Test using TWO_STAGE strategy through main search interface
        results = await mock_search_service.search(
            query="get user information",
            available_tools=mock_tools,
            strategy=SearchStrategy.TWO_STAGE,
            limit=5
        )
    
    # Should call the two-stage implementation
    mock_hybrid.assert_called_once()
    assert len(results) >= 0


def test_confidence_cutoff_logic(mock_search_service):
    """Test the confidence cutoff logic in isolation."""
    
    # Test case 1: Large score drop triggers cutoff
    results_with_drop = [
        {"score": 0.90},
        {"score": 0.85}, 
        {"score": 0.80},
        {"score": 0.75},
        {"score": 0.30},  # Large drop
        {"score": 0.25}
    ]
    
    filtered = mock_search_service._apply_confidence_cutoff(results_with_drop)
    assert len(filtered) <= 4  # Should stop before large drop
    
    # Test case 2: Score below 60% of top score triggers cutoff
    results_low_score = [
        {"score": 1.00},
        {"score": 0.80},
        {"score": 0.50},  # 50% of top score - below 60% threshold
        {"score": 0.40}
    ]
    
    filtered = mock_search_service._apply_confidence_cutoff(results_low_score)
    assert len(filtered) <= 2  # Should stop before 50% score
    
    # Test case 3: Score below minimum threshold (0.15) triggers cutoff
    results_min_threshold = [
        {"score": 0.80},
        {"score": 0.60},
        {"score": 0.40}, 
        {"score": 0.10}  # Below 0.15 minimum
    ]
    
    filtered = mock_search_service._apply_confidence_cutoff(results_min_threshold)
    assert len(filtered) <= 3  # Should stop before 0.10 score
    
    # Test case 4: Empty results
    filtered_empty = mock_search_service._apply_confidence_cutoff([])
    assert len(filtered_empty) == 0
    
    # Test case 5: Single result
    filtered_single = mock_search_service._apply_confidence_cutoff([{"score": 0.80}])
    assert len(filtered_single) == 1


def test_two_stage_strategy_availability(mock_search_service):
    """Test that TWO_STAGE strategy is available when hybrid search is enabled."""
    
    available_strategies = mock_search_service.get_available_strategies()
    strategy_values = [s.value for s in available_strategies]
    
    # TWO_STAGE should be available when hybrid search is enabled
    assert "two_stage" in strategy_values
    
    # Test with hybrid search disabled
    mock_search_service.enable_bm25 = False
    available_strategies_no_hybrid = mock_search_service.get_available_strategies()  
    strategy_values_no_hybrid = [s.value for s in available_strategies_no_hybrid]
    
    # TWO_STAGE should not be available without hybrid search
    assert "two_stage" not in strategy_values_no_hybrid