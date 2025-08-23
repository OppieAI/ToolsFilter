"""Embedding service using LiteLLM for multiple provider support."""

import hashlib
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any
from functools import lru_cache

import redis
from litellm import embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Service for generating embeddings using LiteLLM."""
    
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, batch_size: Optional[int] = None):
        """Initialize embedding service with Redis cache."""
        self.model = model or settings.primary_embedding_model
        self.api_key = api_key or settings.primary_embedding_api_key
        self.batch_size = batch_size or settings.embedding_batch_size
        self._dimension_cache = {}  # Cache dimensions to avoid repeated API calls
        
        # Initialize Redis cache
        try:
            self.redis_client = redis.Redis.from_url(
                settings.redis_url,
                decode_responses=True
            )
            self.redis_client.ping()
            self.cache_enabled = True
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.warning(f"Redis cache initialization failed: {e}. Running without cache.")
            self.redis_client = None
            self.cache_enabled = False
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model combination."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        model_key = model.replace("/", "_").replace("-", "_")
        return f"emb:{model_key}:{text_hash}"
    
    async def _get_from_cache(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache if available."""
        if not self.cache_enabled:
            return None
        
        try:
            key = self._get_cache_key(text, model)
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def _save_to_cache(self, text: str, model: str, embedding: List[float]):
        """Save embedding to cache."""
        if not self.cache_enabled:
            return
        
        try:
            key = self._get_cache_key(text, model)
            self.redis_client.setex(
                key,
                settings.embedding_cache_ttl,
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_embedding(self, text: str, model: str) -> List[float]:
        """Generate embedding using LiteLLM with retry logic."""
        try:
            kwargs = {
                "model": model,
                "input": [text]
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
                
            response = embedding(**kwargs)
            return response.data[0]["embedding"]
        except Exception as e:
            logger.error(f"Embedding generation failed for model {model}: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        cached = await self._get_from_cache(text, self.model)
        if cached:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return cached
        
        # Generate embedding
        embedding_vector = await self._generate_embedding(text, self.model)
        await self._save_to_cache(text, self.model, embedding_vector)
        return embedding_vector
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            texts_to_generate = []
            text_indices = []
            
            for idx, text in enumerate(batch):
                cached = await self._get_from_cache(text, self.model)
                if cached:
                    batch_embeddings.append((idx, cached))
                else:
                    texts_to_generate.append(text)
                    text_indices.append(idx)
            
            # Generate embeddings for uncached texts
            if texts_to_generate:
                kwargs = {
                    "model": self.model,
                    "input": texts_to_generate
                }
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                    
                response = embedding(**kwargs)
                
                for idx, emb_data in zip(text_indices, response.data):
                    embedding_vector = emb_data["embedding"]
                    batch_embeddings.append((idx, embedding_vector))
                    await self._save_to_cache(
                        texts_to_generate[text_indices.index(idx)],
                        self.model,
                        embedding_vector
                    )
            
            # Sort by original index and extract embeddings
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
        
        return embeddings
    
    async def embed_conversation(self, messages: List[Dict[str, Any]]) -> List[float]:
        """
        Generate embedding for a conversation.
        
        Args:
            messages: List of messages in OpenAI/Anthropic format
            
        Returns:
            Single embedding vector representing the conversation
        """
        # Concatenate last N messages for context
        context_messages = messages[-10:]  # Last 10 messages
        
        # Build conversation text
        conversation_parts = []
        for msg in context_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # Handle different content types
            if isinstance(content, str):
                conversation_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Handle multimodal content
                text_parts = [
                    part.get("text", "") 
                    for part in content 
                    if part.get("type") == "text"
                ]
                if text_parts:
                    conversation_parts.append(f"{role}: {' '.join(text_parts)}")
        
        conversation_text = "\n".join(conversation_parts)
        return await self.embed_text(conversation_text)
    
    async def embed_queries(self, queries: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Generate embeddings for multiple query representations in parallel.
        
        Args:
            queries: Dictionary of query names to query texts
            
        Returns:
            Dictionary of query names to embedding vectors
        """
        if not queries:
            return {}
        
        # Create tasks for parallel embedding
        tasks = []
        query_names = []
        
        for name, text in queries.items():
            if text:  # Skip empty queries
                tasks.append(self.embed_text(text))
                query_names.append(name)
        
        # Execute all embeddings in parallel
        if tasks:
            embeddings = await asyncio.gather(*tasks)
            return dict(zip(query_names, embeddings))
        
        return {}
    
    async def embed_tool(self, tool: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for a tool definition.
        
        Args:
            tool: Tool definition in OpenAI function format
            
        Returns:
            Embedding vector for the tool
        """
        # Extract tool information
        if tool.get("type") == "function":
            # Handle both old nested and new flat structure
            if "function" in tool:
                function = tool.get("function", {})
                name = function.get("name", "")
                description = function.get("description", "")
                parameters = function.get("parameters", {})
            else:
                name = tool.get("name", "")
                description = tool.get("description", "")
                parameters = tool.get("parameters", {})
            
            # Build tool text representation
            tool_text = f"{name}: {description}"
            
            # Add parameter information
            if parameters.get("properties"):
                param_desc = []
                for param_name, param_info in parameters["properties"].items():
                    param_type = param_info.get("type", "any")
                    param_desc.append(f"{param_name} ({param_type})")
                
                if param_desc:
                    tool_text += f" | Parameters: {', '.join(param_desc)}"
        else:
            # Fallback for other tool formats
            tool_text = json.dumps(tool)
        
        return await self.embed_text(tool_text)
    
    async def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for a model by generating a test embedding."""
        model = model or self.model
        
        # Check cache first
        if model in self._dimension_cache:
            return self._dimension_cache[model]
        
        try:
            # Generate a simple test embedding to get dimension
            kwargs = {
                "model": model,
                "input": ["test"]
            }
            if self.api_key:
                kwargs["api_key"] = self.api_key
                
            response = embedding(**kwargs)
            dimension = len(response.data[0]["embedding"])
            self._dimension_cache[model] = dimension
            logger.info(f"Detected embedding dimension for {model}: {dimension}")
            return dimension
        except Exception as e:
            logger.error(f"Failed to get embedding dimension for {model}: {e}")
            # Return a sensible default
            return 1536
    
    async def cache_health_check(self) -> bool:
        """Check if cache is healthy."""
        if not self.cache_enabled:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except Exception:
            return False