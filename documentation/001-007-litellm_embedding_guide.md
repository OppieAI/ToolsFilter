# LiteLLM Embedding Integration Guide for PTR

## Overview
LiteLLM provides a unified interface for 100+ LLM providers including all major embedding models.

## Supported Cloud Embedding Providers

### 1. OpenAI
```python
from litellm import embedding

response = embedding(
    model="text-embedding-3-small",  # or text-embedding-3-large, text-embedding-ada-002
    input=["Your text here"],
    api_key="your-openai-key"  # or set OPENAI_API_KEY env var
)
```

### 2. Voyage AI
```python
response = embedding(
    model="voyage-2",  # or voyage-large-2, voyage-code-2
    input=["Your text here"],
    api_key="your-voyage-key"  # or set VOYAGE_API_KEY env var
)
```

### 3. Cohere
```python
response = embedding(
    model="embed-english-v3.0",  # or embed-multilingual-v3.0
    input=["Your text here"],
    api_key="your-cohere-key"  # or set COHERE_API_KEY env var
)
```

### 4. Google (Gemini/Vertex AI)
```python
response = embedding(
    model="textembedding-gecko",  # Google's embedding model
    input=["Your text here"],
    vertex_project="your-project",
    vertex_location="us-central1"
)
```

### 5. Amazon Bedrock
```python
response = embedding(
    model="bedrock/amazon.titan-embed-text-v1",
    input=["Your text here"],
    aws_access_key_id="your-key",
    aws_secret_access_key="your-secret",
    aws_region_name="us-east-1"
)
```

## Installation
```bash
pip install litellm
```

## PTR Integration Example

```python
from fastapi import FastAPI
from litellm import embedding
from qdrant_client import QdrantClient
from typing import List, Dict
import os

app = FastAPI()
qdrant = QdrantClient("localhost", port=6333)

# Configuration - easily switch between providers
EMBEDDING_CONFIG = {
    "provider": "openai",  # or "voyage", "cohere", "google"
    "models": {
        "openai": "text-embedding-3-small",
        "voyage": "voyage-2",
        "cohere": "embed-english-v3.0",
        "google": "textembedding-gecko"
    }
}

class EmbeddingService:
    def __init__(self):
        self.provider = EMBEDDING_CONFIG["provider"]
        self.model = EMBEDDING_CONFIG["models"][self.provider]
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using configured provider"""
        response = embedding(
            model=self.model,
            input=[text]
        )
        return response.data[0]["embedding"]
    
    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        response = embedding(
            model=self.model,
            input=texts
        )
        return [item["embedding"] for item in response.data]
    
    def switch_provider(self, provider: str):
        """Switch embedding provider at runtime"""
        if provider in EMBEDDING_CONFIG["models"]:
            self.provider = provider
            self.model = EMBEDDING_CONFIG["models"][provider]

# Initialize service
embedding_service = EmbeddingService()

@app.post("/embed_tool")
async def embed_tool(tool_name: str, tool_description: str):
    """Generate and store tool embedding"""
    embedding_vector = await embedding_service.generate_embedding(
        f"{tool_name}: {tool_description}"
    )
    
    qdrant.upsert(
        collection_name="tools",
        points=[{
            "id": tool_name,
            "vector": embedding_vector,
            "payload": {"description": tool_description}
        }]
    )
    
    return {
        "status": "success",
        "provider": embedding_service.provider,
        "dimension": len(embedding_vector)
    }

@app.post("/find_similar_tools")
async def find_similar_tools(query: str, limit: int = 10):
    """Find tools similar to query"""
    query_embedding = await embedding_service.generate_embedding(query)
    
    results = qdrant.search(
        collection_name="tools",
        query_vector=query_embedding,
        limit=limit
    )
    
    return {
        "similar_tools": [
            {
                "name": hit.id,
                "score": hit.score,
                "description": hit.payload["description"]
            }
            for hit in results
        ]
    }

@app.post("/switch_embedding_provider")
async def switch_provider(provider: str):
    """Switch embedding provider on the fly"""
    embedding_service.switch_provider(provider)
    return {"new_provider": provider, "model": embedding_service.model}
```

## Environment Variables Setup
```bash
# .env file
OPENAI_API_KEY=sk-...
VOYAGE_API_KEY=pa-...
COHERE_API_KEY=...
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# For Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1
```

## Cost Comparison (per 1M tokens)
- OpenAI text-embedding-3-small: $0.02
- OpenAI text-embedding-3-large: $0.13
- Voyage-2: $0.02
- Cohere embed-v3: $0.10
- Google textembedding-gecko: $0.025

## Performance Tips

1. **Batch Requests**: LiteLLM automatically batches when possible
```python
# Good - single API call
embeddings = embedding(model="voyage-2", input=["text1", "text2", "text3"])

# Bad - multiple API calls
for text in texts:
    embedding(model="voyage-2", input=[text])
```

2. **Caching**: Implement Redis caching for repeated embeddings
```python
import hashlib
import json
import redis

redis_client = redis.Redis()

async def get_embedding_with_cache(text: str, model: str):
    # Generate cache key
    cache_key = f"emb:{model}:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Generate embedding
    response = embedding(model=model, input=[text])
    embedding_vector = response.data[0]["embedding"]
    
    # Cache for 24 hours
    redis_client.setex(cache_key, 86400, json.dumps(embedding_vector))
    
    return embedding_vector
```

3. **Error Handling**: LiteLLM provides automatic retries
```python
from litellm import embedding, RateLimitError, APIError

try:
    response = embedding(model="voyage-2", input=texts)
except RateLimitError:
    # Wait and retry or switch provider
    embedding_service.switch_provider("openai")
    response = embedding(model="text-embedding-3-small", input=texts)
except APIError as e:
    # Log and handle error
    print(f"API Error: {e}")
```

## A/B Testing Different Providers
```python
import random

@app.post("/embed_with_ab_test")
async def embed_with_ab_test(text: str):
    # Randomly select provider for A/B testing
    providers = ["openai", "voyage", "cohere"]
    selected_provider = random.choice(providers)
    
    embedding_service.switch_provider(selected_provider)
    embedding_vector = await embedding_service.generate_embedding(text)
    
    # Log for analysis
    print(f"Used {selected_provider} for embedding")
    
    return {
        "embedding": embedding_vector,
        "provider_used": selected_provider
    }
```

## Why LiteLLM for PTR?
1. **No vendor lock-in**: Switch providers with one line
2. **Cost optimization**: Use cheaper providers for less critical embeddings
3. **Reliability**: Automatic fallback to other providers
4. **Simple integration**: Unified API for all providers
5. **Production ready**: Used by major companies

This setup gives you maximum flexibility to experiment with different embedding providers and optimize for cost/performance without changing your code.