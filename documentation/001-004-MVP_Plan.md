# MVP Implementation Plan - PTR Tool Filter

## Quick Start Guide

### Week 1: Foundation Setup
1. **Core Infrastructure**
   - FastAPI for async REST API
   - Qdrant for vector storage (Docker deployment)
   - Redis for embedding cache
   - LiteLLM for unified embedding access

2. **Embedding Strategy**
   - Start with Voyage AI (voyage-2) via LiteLLM
   - Implement OpenAI fallback (text-embedding-3-small)
   - Cache embeddings in Redis with 1hr TTL

3. **Initial Algorithm**
   - Simple semantic matching only (skip PTR stages initially)
   - Cosine similarity with configurable threshold
   - Return top-10 tools ranked by score

### Finalized Tech Stack
```yaml
# Core Services
API Framework: FastAPI 0.104+
Vector Database: Qdrant 1.7+
Cache: Redis 7.2+
Language: Python 3.11+

# Embedding & AI
Embedding Service: LiteLLM
Primary Model: voyage-2
Fallback Model: text-embedding-3-small

# Evaluation & Monitoring
Metrics: RAGAS
Observability: Phoenix (Arize)
Experiments: MLflow

# Development Tools
Testing: pytest + pytest-asyncio
Linting: ruff
Type Checking: mypy
API Docs: OpenAPI/Swagger (auto-generated)

# Deployment
Containerization: Docker + Docker Compose
Orchestration: Kubernetes (Phase 2)
```

### MVP Features (Week 1)
- [x] Parse Claude/OpenAI message formats
- [x] Generate embeddings via LiteLLM
- [x] Store and search embeddings in Qdrant
- [x] Return top-K tools based on similarity
- [x] Single REST endpoint with OpenAPI docs
- [x] Basic evaluation with RAGAS

## 🚀 MVP Implementation Tasks (Week 1)

### Day 1-2: Project Setup
- [x] Create GitHub repository
- [x] Set up project structure as per MVP plan
- [x] Create requirements.txt with dependencies
- [x] Set up Docker Compose with Qdrant and Redis
- [x] Create .env.example with required keys
- [x] Initialize FastAPI application

### Day 3-4: Core Services
- [x] Implement LiteLLM embedding service wrapper
- [x] Create Qdrant client and collections
- [x] Implement Redis caching layer
- [x] Build message parsers (Claude/OpenAI formats)
- [x] Create tool registry loader for MCP tools

### Day 5-6: API & Evaluation
- [x] Implement `/api/v1/tools/filter` endpoint
- [x] Add request/response validation with Pydantic
- [x] Create sample tool dataset for testing
- [x] Integrate RAGAS for basic evaluation
- [x] Write initial test suite with pytest

### Day 7: Documentation & Testing
- [x] Generate OpenAPI documentation
- [x] Create README with setup instructions
- [x] Run load tests to verify <100ms latency
- [x] Create sample client code
- [x] Deploy MVP with Docker Compose


### Project Structure
```
ToolsFilter/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app
│   │   └── endpoints.py     # API routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py        # Settings
│   │   └── models.py        # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # LiteLLM integration
│   │   ├── vector_store.py  # Qdrant client
│   │   └── message_parser.py # Message parsing
│   └── evaluation/
│       ├── __init__.py
│       └── metrics.py       # RAGAS integration
├── tests/
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

### First Sprint Deliverables
1. **API Endpoint**: `POST /api/v1/tools/filter`
2. **Message Parser**: Claude + OpenAI format support
3. **Embedding Service**: LiteLLM with caching
4. **Vector Store**: Qdrant with tool embeddings
5. **Basic Evaluation**: Precision/Recall metrics
6. **Documentation**: OpenAPI specs + README

### Success Metrics
- API latency < 100ms (P95)
- Precision@10 > 60%
- All tests passing
- Docker Compose deployment working

This MVP provides a solid foundation that can be enhanced with PTR stages, historical learning, and advanced features in subsequent phases.
