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
from src.services.message_parser import MessageParser

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


@router.post("/tools/filter", response_model=ToolFilterResponse)
async def filter_tools(
    request: ToolFilterRequest,
    authenticated: bool = Depends(verify_api_key),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    fallback_embedding_service: EmbeddingService = Depends(get_fallback_embedding_service),
    fallback_vector_store: VectorStoreService = Depends(get_fallback_vector_store)
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
            tool_name = tool.function.name if tool.type == "function" else tool.dict().get("name", "unknown")
            
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
            if tool.type == "function" and tool.function:
                available_tool_names.append(tool.function.name)
            else:
                # Fallback for other tool types
                tool_dict = tool.dict() if hasattr(tool, 'dict') else tool
                available_tool_names.append(tool_dict.get("name", "unknown"))
        
        # Search for similar tools with automatic fallback
        max_tools = request.max_tools or settings.max_tools_to_return
        model_used = embedding_service.model
        
        try:
            # Try primary store first
            conversation_embedding = await embedding_service.embed_conversation(request.messages)
            similar_tools = await vector_store.search_similar_tools(
                query_embedding=conversation_embedding,
                limit=max_tools * 2,  # Get more candidates for filtering
                score_threshold=settings.primary_similarity_threshold,
                filter_dict={"name": available_tool_names}  # Only search within available tools
            )
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            
            # Try fallback if available
            if fallback_vector_store and fallback_embedding_service:
                logger.info("Attempting search with fallback model...")
                conversation_embedding = await fallback_embedding_service.embed_conversation(request.messages)
                similar_tools = await fallback_vector_store.search_similar_tools(
                    query_embedding=conversation_embedding,
                    limit=max_tools * 2,
                    score_threshold=settings.fallback_similarity_threshold,
                    filter_dict={"name": available_tool_names}  # Only search within available tools
                )
                model_used = fallback_embedding_service.model
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
        
        response = ToolFilterResponse(
            recommended_tools=recommended_tools,
            metadata={
                "processing_time_ms": round(processing_time, 2),
                "embedding_model": model_used,
                "total_tools_analyzed": len(request.available_tools),
                "conversation_messages": len(request.messages),
                "request_id": request_id,
                "conversation_patterns": conversation_analysis["topics"]
            }
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
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    fallback_embedding_service: EmbeddingService = Depends(get_fallback_embedding_service),
    fallback_vector_store: VectorStoreService = Depends(get_fallback_vector_store)
):
    """
    Search for tools by text query.
    
    This endpoint allows searching for tools using natural language.
    """
    try:
        model_used = embedding_service.model
        
        try:
            # Try primary store first
            query_embedding = await embedding_service.embed_text(query)
            similar_tools = await vector_store.search_similar_tools(
                query_embedding=query_embedding,
                limit=limit
            )
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            
            # Try fallback if available
            if fallback_vector_store and fallback_embedding_service:
                logger.info("Attempting search with fallback model...")
                query_embedding = await fallback_embedding_service.embed_text(query)
                similar_tools = await fallback_vector_store.search_similar_tools(
                    query_embedding=query_embedding,
                    limit=limit
                )
                model_used = fallback_embedding_service.model
            else:
                raise
        
        return {
            "query": query,
            "results": similar_tools,
            "model_used": model_used
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


def _tool_to_text(tool: Dict[str, Any]) -> str:
    """Convert tool to text for embedding."""
    if tool.get("type") == "function":
        function = tool.get("function", {})
        name = function.get("name", "")
        description = function.get("description", "")
        return f"{name}: {description}"
    return str(tool)


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