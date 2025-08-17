# Qdrant Setup Guide for PTR Tool Filter

## Quick Start

### 1. Local Development Setup

#### Using Docker (Recommended)
```bash
# Single instance for development
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

#### Using Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    restart: always
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    environment:
      - QDRANT__LOG_LEVEL=INFO
      - QDRANT__SERVICE__HTTP_PORT=6333
```

### 2. Python Client Setup

```bash
pip install qdrant-client fastembed
```

### 3. Basic Integration Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

# Initialize client
client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=True  # Use gRPC for better performance
)

# Create collection for tool embeddings
client.create_collection(
    collection_name="tools",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

# Index tool embeddings
def index_tool(tool_id: str, tool_description: str, embedding: list[float]):
    client.upsert(
        collection_name="tools",
        points=[
            PointStruct(
                id=tool_id,
                vector=embedding,
                payload={
                    "description": tool_description,
                    "name": tool_id,
                    "usage_count": 0
                }
            )
        ]
    )

# Search for similar tools
def find_similar_tools(query_embedding: list[float], limit: int = 10):
    results = client.search(
        collection_name="tools",
        query_vector=query_embedding,
        limit=limit,
        with_payload=True
    )
    return results
```

## Production Setup

### 1. Cluster Configuration

Create `qdrant-cluster.yml`:
```yaml
version: '3.8'

services:
  qdrant-node-1:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./storage/node1:/qdrant/storage:z
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__CONSENSUS__TICK_PERIOD_MS=100
    command: ["./qdrant", "--bootstrap", "http://qdrant-node-1:6335"]

  qdrant-node-2:
    image: qdrant/qdrant:latest
    ports:
      - "6336:6333"
    volumes:
      - ./storage/node2:/qdrant/storage:z
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
    command: ["./qdrant", "--uri", "http://qdrant-node-2:6335", "--bootstrap", "http://qdrant-node-1:6335"]

  qdrant-node-3:
    image: qdrant/qdrant:latest
    ports:
      - "6339:6333"
    volumes:
      - ./storage/node3:/qdrant/storage:z
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
    command: ["./qdrant", "--uri", "http://qdrant-node-3:6335", "--bootstrap", "http://qdrant-node-1:6335"]
```

### 2. Collection Configuration for PTR

```python
# Create collections with sharding and replication
client.create_collection(
    collection_name="tools",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=3,  # Distribute across nodes
    replication_factor=2,  # Redundancy
    write_consistency_factor=2  # Strong consistency
)

client.create_collection(
    collection_name="conversation_embeddings",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    shard_number=3,
    replication_factor=2,
    on_disk_payload=True  # Store payload on disk for large conversations
)

client.create_collection(
    collection_name="historical_patterns",
    vectors_config={
        "query": VectorParams(size=1536, distance=Distance.COSINE),
        "tool_bundle": VectorParams(size=768, distance=Distance.EUCLID)
    },
    shard_number=3,
    replication_factor=2
)
```

### 3. Performance Optimization

```python
# Enable indexing optimization
client.update_collection(
    collection_name="tools",
    optimizer_config={
        "indexing_threshold": 10000,
        "flush_interval_sec": 30,
        "max_optimization_threads": 4
    }
)

# Use quantization for memory efficiency
from qdrant_client.models import QuantizationConfig, ScalarQuantization

client.update_collection(
    collection_name="tools",
    quantization_config=ScalarQuantization(
        scalar={
            "type": "int8",
            "quantile": 0.99,
            "always_ram": True
        }
    )
)
```

## Integration with PTR

### 1. Tool Indexing Service

```python
from typing import List, Dict
import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

class ToolIndexer:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = AsyncQdrantClient(url=qdrant_url, prefer_grpc=True)
        
    async def index_tools(self, tools: List[Dict]):
        """Index MCP tool definitions into Qdrant"""
        points = []
        for tool in tools:
            embedding = await self.generate_embedding(tool['description'])
            points.append(
                PointStruct(
                    id=tool['name'],
                    vector=embedding,
                    payload={
                        "name": tool['name'],
                        "description": tool['description'],
                        "parameters": tool.get('parameters', {}),
                        "category": tool.get('category', 'general')
                    }
                )
            )
        
        await self.client.upsert(
            collection_name="tools",
            points=points,
            wait=True
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        # Use your embedding service here
        pass
```

### 2. Similarity Search for PTR

```python
class PTRSearcher:
    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self.client = AsyncQdrantClient(url=qdrant_url, prefer_grpc=True)
    
    async def find_relevant_tools(
        self, 
        conversation_embedding: List[float], 
        limit: int = 20,
        score_threshold: float = 0.7
    ):
        """Find tools relevant to conversation context"""
        results = await self.client.search(
            collection_name="tools",
            query_vector=conversation_embedding,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True
        )
        
        return [
            {
                "tool_name": hit.payload["name"],
                "score": hit.score,
                "description": hit.payload["description"]
            }
            for hit in results
        ]
```

## Monitoring and Maintenance

### 1. Health Check
```python
# Check cluster health
health = client.get_cluster_info()
print(f"Nodes: {len(health.nodes)}")
print(f"Status: {health.status}")
```

### 2. Backup Strategy
```bash
# Snapshot creation
curl -X POST 'http://localhost:6333/collections/tools/snapshots'

# Download snapshot
curl -X GET 'http://localhost:6333/collections/tools/snapshots/{snapshot_name}' \
     --output tools_backup.snapshot
```

### 3. Performance Monitoring
```python
# Get collection info
info = client.get_collection("tools")
print(f"Vectors count: {info.vectors_count}")
print(f"Indexed vectors: {info.indexed_vectors_count}")
print(f"Memory usage: {info.segments[0].ram_usage_bytes / 1024 / 1024:.2f} MB")
```

## Best Practices for PTR

1. **Pre-compute tool embeddings** during deployment
2. **Use batch operations** for better performance
3. **Enable quantization** for large tool sets (>10k tools)
4. **Implement caching** for frequently accessed tools
5. **Monitor search latencies** and adjust indexing parameters
6. **Use gRPC** for production deployments
7. **Set up proper backups** before production deployment