"""API endpoints for PTR Tool Filter."""

import time
import logging
from typing import List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader

from src.core.config import get_settings
from src.core.models import (
    ToolFilterRequest,
    ToolFilterResponse,
    RecommendedTool,
    Tool
)
from src.services.embeddings import EmbeddingService
from src.services.vector_store import VectorStoreService
from src.services.search_service import SearchService, SearchStrategy
from src.services.search_pipeline_config import (
    SearchPipelineConfig, get_production_config
)
from src.services.message_parser import MessageParser
from src.services.embedding_enhancer import ToolEmbeddingEnhancer
from src.services.query_enhancer import QueryEnhancer

logger = logging.getLogger(__name__)
settings = get_settings()

# Create router
router = APIRouter()

# Optional API key authentication
api_key_header = APIKeyHeader(
    name=settings.api_key_header,
    auto_error=False
)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key if enabled."""
    if not settings.enable_api_key:
        return True
    
    if not api_key:
        raise HTTPException(status_code=403, detail="API key required")
    
    if api_key not in settings.api_keys_list:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    from src.api.main import get_embedding_service as get_service
    return get_service()


def get_vector_store() -> VectorStoreService:
    """Get vector store service instance."""
    from src.api.main import get_vector_store as get_service
    return get_service()


def get_fallback_embedding_service() -> EmbeddingService:
    """Get fallback embedding service instance."""
    from src.api.main import get_fallback_embedding_service as get_service
    return get_service()


def get_fallback_vector_store() -> VectorStoreService:
    """Get fallback vector store service instance."""
    from src.api.main import get_fallback_vector_store as get_service
    return get_service()


def get_search_service() -> SearchService:
    """Get search service instance."""
    from src.api.main import get_search_service as get_service
    return get_service()


def get_fallback_search_service() -> SearchService:
    """Get fallback search service instance."""
    from src.api.main import get_fallback_search_service as get_service
    return get_service()


def _get_production_pipeline_config() -> SearchPipelineConfig:
    """
    Create production pipeline configuration based on current settings.
    
    This ensures the API uses the same pipeline as training and evaluation,
    eliminating training-production pipeline mismatches.
    """
    # Determine enabled features based on settings
    enable_ltr = getattr(settings, 'enable_ltr', True)
    enable_cross_encoder = getattr(settings, 'enable_cross_encoder', True) 
    enable_hybrid = getattr(settings, 'enable_hybrid_search', True)
    enable_two_stage = getattr(settings, 'enable_two_stage_filtering', False)
    
    # Use two-stage config if explicitly enabled
    if enable_two_stage:
        from src.services.search_pipeline_config import get_two_stage_config
        return get_two_stage_config(
            stage1_threshold=getattr(settings, 'two_stage_stage1_threshold', 0.10),
            stage1_limit=getattr(settings, 'two_stage_stage1_limit', 50),
            stage2_threshold=getattr(settings, 'two_stage_stage2_threshold', 0.15),
            enable_confidence_cutoff=getattr(settings, 'two_stage_enable_confidence_cutoff', True)
        )
    
    # Use production config with appropriate feature flags
    return get_production_config(
        enable_ltr=enable_ltr,
        enable_cross_encoder=enable_cross_encoder,
        enable_bm25=enable_hybrid,  # BM25 is part of hybrid search
        final_threshold=settings.primary_similarity_threshold,
        final_limit=settings.max_tools_to_return,
        semantic_weight=getattr(settings, 'semantic_weight', 0.7),
        bm25_weight=getattr(settings, 'bm25_weight', 0.3),
        cross_encoder_top_k=getattr(settings, 'cross_encoder_top_k', 30),
        enable_confidence_cutoff=True  # Production should use confidence cutoffs
    )


@router.post("/tools/filter", response_model=ToolFilterResponse)
async def filter_tools(
    request: ToolFilterRequest,
    authenticated: bool = Depends(verify_api_key),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    search_service: SearchService = Depends(get_search_service),
    fallback_embedding_service: EmbeddingService = Depends(get_fallback_embedding_service),
    fallback_vector_store: VectorStoreService = Depends(get_fallback_vector_store),
    fallback_search_service: SearchService = Depends(get_fallback_search_service)
):
    """
    Filter tools based on conversation context.
    
    This endpoint analyzes the conversation messages and returns the most
    relevant tools from the available tool set.
    """
    start_time = time.time()
    request_id = str(uuid4())
    
    try:
        # Parse messages
        parser = MessageParser()
        conversation_text, used_tools = parser.extract_conversation_context(request.messages)
        user_intent = parser.extract_user_intent(request.messages)
        conversation_analysis = parser.analyze_conversation_pattern(request.messages)
        
        logger.info(f"Request {request_id}: Analyzing conversation with {len(request.messages)} messages")
        
        # Index available tools if not already indexed
        # In production, this would be done separately during tool registration
        tools_to_index = []
        tool_texts = []
        
        for tool in request.available_tools:
            tool_name = tool.name if hasattr(tool, 'name') else tool.dict().get("name", "unknown")
            
            # Check if tool already exists in primary store
            existing_tool = await vector_store.get_tool_by_name(tool_name)
            if not existing_tool:
                tools_to_index.append(tool.dict())
                tool_texts.append(_tool_to_text(tool.dict()))
        
        # Index new tools in both stores if needed
        if tools_to_index:
            try:
                # Index in primary store
                primary_embeddings = await embedding_service.embed_batch(tool_texts)
                await vector_store.index_tools_batch(tools_to_index, primary_embeddings)
                logger.info(f"Indexed {len(tools_to_index)} tools in primary store")
            except Exception as e:
                logger.error(f"Failed to index in primary store: {e}")
            
            # Also index in fallback store if available
            if fallback_vector_store and fallback_embedding_service:
                try:
                    fallback_embeddings = await fallback_embedding_service.embed_batch(tool_texts)
                    await fallback_vector_store.index_tools_batch(tools_to_index, fallback_embeddings)
                    logger.info(f"Indexed {len(tools_to_index)} tools in fallback store")
                except Exception as e:
                    logger.error(f"Failed to index in fallback store: {e}")
        
        # Extract available tool names for filtering
        available_tool_names = []
        for tool in request.available_tools:
            if hasattr(tool, 'name'):
                available_tool_names.append(tool.name)
            else:
                # Fallback for other tool types
                tool_dict = tool.dict() if hasattr(tool, 'dict') else tool
                available_tool_names.append(tool_dict.get("name", "unknown"))
        
        # Get production pipeline configuration
        max_tools = request.max_tools or settings.max_tools_to_return
        pipeline_config = _get_production_pipeline_config()
        
        # Override final_limit with request-specific limit
        pipeline_config.final_limit = max_tools
        
        logger.info(f"Using production pipeline: LTR={pipeline_config.enable_ltr}, "
                   f"CrossEncoder={pipeline_config.enable_cross_encoder}, "
                   f"Hybrid={pipeline_config.enable_bm25}")
        
        try:
            # Check if query enhancement is enabled and semantic-only pipeline
            if (getattr(settings, 'enable_query_enhancement', True) and 
                not pipeline_config.enable_bm25 and not pipeline_config.enable_cross_encoder and not pipeline_config.enable_ltr):
                # Special case: multi-query search with query enhancement for semantic-only
                query_enhancer = QueryEnhancer()
                enhanced_query = query_enhancer.enhance_query(request.messages, used_tools)
                
                # Generate embeddings for enhanced queries
                query_embeddings = await search_service.embedding_service.embed_queries(enhanced_query["queries"])
                
                # Use search_multi_query method
                similar_tools = await search_service.search_multi_query(
                    query_embeddings=query_embeddings,
                    weights=enhanced_query["weights"],
                    limit=max_tools * 2,
                    score_threshold=settings.primary_similarity_threshold,
                    filter_dict={"name": available_tool_names}
                )
                
                # Store metadata
                conversation_analysis.update(enhanced_query["metadata"])
                conversation_analysis["search_method"] = "multi_query_enhanced"
            else:
                # Use unified search pipeline with production configuration
                similar_tools = await search_service.search_with_config(
                    messages=request.messages,
                    available_tools=request.available_tools,
                    config=pipeline_config
                )
                
                # Add metadata based on pipeline configuration
                conversation_analysis["search_method"] = "production_pipeline"
                conversation_analysis["pipeline_config"] = {
                    "enable_ltr": pipeline_config.enable_ltr,
                    "enable_cross_encoder": pipeline_config.enable_cross_encoder,
                    "enable_hybrid": pipeline_config.enable_bm25,
                    "final_threshold": pipeline_config.final_threshold,
                    "final_limit": pipeline_config.final_limit
                }
                if pipeline_config.enable_bm25:
                    conversation_analysis["hybrid_method"] = "weighted"
                    conversation_analysis["semantic_weight"] = pipeline_config.semantic_weight
                    conversation_analysis["bm25_weight"] = pipeline_config.bm25_weight
            
            model_used = search_service.embedding_service.model
            
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            
            # Try fallback if available
            if fallback_search_service:
                logger.info("Attempting search with fallback model...")
                
                try:
                    # Use same pipeline config with fallback service
                    fallback_config = pipeline_config
                    fallback_config.final_threshold = settings.fallback_similarity_threshold
                    
                    similar_tools = await fallback_search_service.search_with_config(
                        messages=request.messages,
                        available_tools=request.available_tools,
                        config=fallback_config
                    )
                    
                    conversation_analysis["search_method"] = "production_pipeline_fallback"
                    model_used = fallback_search_service.embedding_service.model
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback search also failed: {fallback_error}")
                    raise
            else:
                raise
        
        # Build response
        recommended_tools = []
        seen_tools = set()
        
        for tool_result in similar_tools:
            tool_name = tool_result["tool_name"]
            
            # Avoid duplicates
            if tool_name in seen_tools:
                continue
            seen_tools.add(tool_name)
            
            # Generate reasoning if requested
            reasoning = None
            if request.include_reasoning:
                reasoning = _generate_reasoning(
                    tool_result,
                    conversation_analysis,
                    user_intent,
                    used_tools
                )
            
            recommended_tools.append(
                RecommendedTool(
                    tool_name=tool_name,
                    confidence=tool_result["score"],
                    reasoning=reasoning
                )
            )
            
            if len(recommended_tools) >= max_tools:
                break
        
        # Calculate metadata
        processing_time = (time.time() - start_time) * 1000
        
        # Build metadata including search method info
        metadata = {
            "processing_time_ms": round(processing_time, 2),
            "embedding_model": model_used,
            "total_tools_analyzed": len(request.available_tools),
            "conversation_messages": len(request.messages),
            "request_id": request_id,
            "conversation_patterns": conversation_analysis["topics"]
        }
        
        # Add search method and pipeline information
        if "search_method" in conversation_analysis:
            metadata["search_method"] = conversation_analysis["search_method"]
            
            # Add pipeline configuration details to metadata
            if "pipeline_config" in conversation_analysis:
                pipeline_info = conversation_analysis["pipeline_config"]
                metadata.update({
                    "pipeline_ltr_enabled": pipeline_info.get("enable_ltr", False),
                    "pipeline_cross_encoder_enabled": pipeline_info.get("enable_cross_encoder", False), 
                    "pipeline_hybrid_enabled": pipeline_info.get("enable_hybrid", False),
                    "pipeline_final_threshold": pipeline_info.get("final_threshold", 0.13),
                    "pipeline_final_limit": pipeline_info.get("final_limit", 10)
                })
                
                if pipeline_info.get("enable_hybrid", False):
                    metadata["semantic_weight"] = conversation_analysis.get("semantic_weight", settings.semantic_weight)
                    metadata["bm25_weight"] = conversation_analysis.get("bm25_weight", settings.bm25_weight)
        
        response = ToolFilterResponse(
            recommended_tools=recommended_tools,
            metadata=metadata
        )
        
        logger.info(
            f"Request {request_id}: Recommended {len(recommended_tools)} tools "
            f"in {processing_time:.2f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to filter tools: {str(e)}"
        )


@router.post("/tools/register")
async def register_tools(
    tools: List[Tool],
    authenticated: bool = Depends(verify_api_key),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    fallback_embedding_service: EmbeddingService = Depends(get_fallback_embedding_service),
    fallback_vector_store: VectorStoreService = Depends(get_fallback_vector_store)
):
    """
    Register new tools in the system.
    
    This endpoint indexes the provided tools for future filtering in both primary and fallback stores.
    """
    try:
        # Prepare tool data
        tool_dicts = [tool.dict() for tool in tools]
        tool_texts = [_tool_to_text(tool.dict()) for tool in tools]
        
        results = {"primary": None, "fallback": None}
        
        # Index in primary store
        try:
            primary_embeddings = await embedding_service.embed_batch(tool_texts)
            primary_ids = await vector_store.index_tools_batch(tool_dicts, primary_embeddings)
            results["primary"] = {
                "success": True,
                "indexed": len(primary_ids),
                "model": embedding_service.model
            }
        except Exception as e:
            logger.error(f"Failed to index in primary store: {e}")
            results["primary"] = {
                "success": False,
                "error": str(e),
                "model": embedding_service.model
            }
        
        # Index in fallback store if available
        if fallback_vector_store and fallback_embedding_service:
            try:
                fallback_embeddings = await fallback_embedding_service.embed_batch(tool_texts)
                fallback_ids = await fallback_vector_store.index_tools_batch(tool_dicts, fallback_embeddings)
                results["fallback"] = {
                    "success": True,
                    "indexed": len(fallback_ids),
                    "model": fallback_embedding_service.model
                }
            except Exception as e:
                logger.error(f"Failed to index in fallback store: {e}")
                results["fallback"] = {
                    "success": False,
                    "error": str(e),
                    "model": fallback_embedding_service.model
                }
        
        # Check if at least one store succeeded
        primary_success = results["primary"] and results["primary"]["success"]
        fallback_success = results["fallback"] and results["fallback"]["success"]
        
        if not primary_success and not fallback_success:
            raise Exception("Failed to index tools in both primary and fallback stores")
        
        return {
            "message": f"Successfully registered {len(tools)} tools",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Failed to register tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register tools: {str(e)}"
        )


@router.get("/tools/search")
async def search_tools(
    query: str,
    limit: int = 10,
    authenticated: bool = Depends(verify_api_key),
    search_service: SearchService = Depends(get_search_service),
    fallback_search_service: SearchService = Depends(get_fallback_search_service)
):
    """
    Search for tools by text query.
    
    This endpoint allows searching for tools using natural language.
    Uses the same production pipeline as the main filter endpoint.
    """
    try:
        # Get production pipeline configuration
        pipeline_config = _get_production_pipeline_config()
        
        # Override final_limit with request-specific limit
        pipeline_config.final_limit = limit
        
        # Determine effective search method based on pipeline config
        if pipeline_config.enable_ltr:
            effective_strategy = "hybrid_ltr_pipeline"
        elif pipeline_config.enable_cross_encoder and pipeline_config.enable_bm25:
            effective_strategy = "hybrid_cross_encoder_pipeline" 
        elif pipeline_config.enable_bm25:
            effective_strategy = "hybrid_pipeline"
        else:
            effective_strategy = "semantic_pipeline"
        
        try:
            # Try primary search service first with production pipeline
            similar_tools = await search_service.search_with_config(
                query=query,
                config=pipeline_config
            )
            model_used = search_service.embedding_service.model
                
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            
            # Try fallback if available
            if fallback_search_service:
                logger.info("Attempting search with fallback model...")
                
                # Use same pipeline config with fallback threshold
                fallback_config = pipeline_config
                fallback_config.final_threshold = settings.fallback_similarity_threshold
                
                similar_tools = await fallback_search_service.search_with_config(
                    query=query,
                    config=fallback_config
                )
                model_used = fallback_search_service.embedding_service.model
                effective_strategy = f"{effective_strategy}_fallback"
            else:
                raise
        
        return {
            "query": query,
            "results": similar_tools,
            "model_used": model_used,
            "search_strategy": effective_strategy,
            "pipeline_config": {
                "enable_ltr": pipeline_config.enable_ltr,
                "enable_cross_encoder": pipeline_config.enable_cross_encoder,
                "enable_hybrid": pipeline_config.enable_bm25,
                "final_threshold": pipeline_config.final_threshold,
                "final_limit": pipeline_config.final_limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to search tools: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search tools: {str(e)}"
        )


@router.get("/tools/info")
async def get_tools_info(
    authenticated: bool = Depends(verify_api_key),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Get information about indexed tools."""
    try:
        collection_info = await vector_store.get_collection_info()
        return {
            "total_tools": collection_info["vectors_count"],
            "indexed_tools": collection_info["indexed_vectors_count"],
            "collection_info": collection_info
        }
    except Exception as e:
        logger.error(f"Failed to get tools info: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tools info: {str(e)}"
        )


@router.get("/collections")
async def list_collections(
    authenticated: bool = Depends(verify_api_key),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """List all vector store collections with their metadata."""
    try:
        collections = await vector_store.list_collections()
        return {
            "collections": collections,
            "total": len(collections)
        }
    except Exception as e:
        logger.error(f"Failed to list collections: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list collections: {str(e)}"
        )


@router.get("/performance")
async def get_performance_stats(
    authenticated: bool = Depends(verify_api_key),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Get performance statistics for search optimization and caching."""
    try:
        stats = await vector_store.get_performance_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get performance stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance stats: {str(e)}"
        )


@router.post("/cache/clear")
async def clear_cache(
    authenticated: bool = Depends(verify_api_key),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """Clear the search result cache."""
    try:
        await vector_store.clear_cache()
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/optimize")
async def optimize_collection(
    target_mode: str = "balanced",
    authenticated: bool = Depends(verify_api_key),
    vector_store: VectorStoreService = Depends(get_vector_store)
):
    """
    Optimize the collection configuration for better performance.
    
    Target modes:
    - "speed": Optimize for fast retrieval
    - "accuracy": Optimize for high accuracy
    - "balanced": Balance between speed and accuracy
    """
    if target_mode not in ["speed", "accuracy", "balanced"]:
        raise HTTPException(
            status_code=400,
            detail="target_mode must be one of: speed, accuracy, balanced"
        )
    
    try:
        success = await vector_store.optimize_collection(target_mode)
        if success:
            return {
                "status": "success",
                "message": f"Collection optimized for {target_mode} mode"
            }
        else:
            return {
                "status": "failed",
                "message": "Collection optimization failed"
            }
    except Exception as e:
        logger.error(f"Failed to optimize collection: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimize collection: {str(e)}"
        )


def _tool_to_text(tool: Dict[str, Any]) -> str:
    """Convert tool to text for embedding using enhanced representation."""
    enhancer = ToolEmbeddingEnhancer()
    return enhancer.tool_to_rich_text(tool)


def _generate_reasoning(
    tool_result: Dict[str, Any],
    conversation_analysis: Dict[str, Any],
    user_intent: str,
    used_tools: List[str]
) -> str:
    """Generate reasoning for tool recommendation."""
    reasons = []
    
    # High similarity score
    if tool_result["score"] > 0.9:
        reasons.append("Very high relevance to the conversation context")
    elif tool_result["score"] > 0.8:
        reasons.append("High relevance to the conversation context")
    
    # Pattern matching
    tool_name = tool_result["tool_name"].lower()
    tool_desc = tool_result.get("description", "").lower()
    
    if conversation_analysis["has_search_intent"] and any(
        keyword in tool_name or keyword in tool_desc 
        for keyword in ["search", "find", "grep", "locate"]
    ):
        reasons.append("Matches search intent in conversation")
    
    if conversation_analysis["has_file_operation"] and any(
        keyword in tool_name or keyword in tool_desc 
        for keyword in ["file", "directory", "path", "fs"]
    ):
        reasons.append("Relevant for file operations discussed")
    
    if conversation_analysis["has_code"] and any(
        keyword in tool_name or keyword in tool_desc 
        for keyword in ["code", "syntax", "parse", "compile"]
    ):
        reasons.append("Useful for code-related tasks")
    
    if conversation_analysis["has_error"] and any(
        keyword in tool_name or keyword in tool_desc 
        for keyword in ["debug", "error", "log", "trace"]
    ):
        reasons.append("Can help with debugging/error handling")
    
    # Historical usage
    if tool_name in used_tools:
        reasons.append("Previously used in this conversation")
    
    # Default reasoning
    if not reasons:
        reasons.append(f"Relevant based on: {tool_result.get('description', 'tool capabilities')}")
    
    return "; ".join(reasons)