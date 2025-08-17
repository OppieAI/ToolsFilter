# PTR Tool Filter

A Precision-driven Tool Recommendation (PTR) system for filtering MCP (Model Context Protocol) tools based on conversation context using semantic search.

## Features

- 🚀 Fast semantic search using vector embeddings
- 🔧 Support for multiple embedding providers (Voyage AI, OpenAI, Cohere)
- 💾 Intelligent caching with Redis
- 🎯 High-performance vector search with Qdrant
- 📊 Built-in evaluation metrics with RAGAS
- 🔄 Compatible with Claude and OpenAI message formats
- 🔀 Model-specific collections to prevent embedding dimension conflicts
- 📝 Collection metadata tracking for model versioning
- 🔁 Automatic fallback to secondary embedding model on failures

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API keys for embedding providers (Voyage AI, OpenAI, or Cohere)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ToolsFilter.git
cd ToolsFilter
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment variables:
```bash
cp .env.example .env
```

5. Edit `.env` and add your API keys:
```env
# Embedding Service Keys (at least one required)
VOYAGE_API_KEY=your_voyage_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional fallback
COHERE_API_KEY=your_cohere_api_key  # Optional

# Important: Include provider prefix in model names
PRIMARY_EMBEDDING_MODEL=voyage/voyage-2
FALLBACK_EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Running the Services

#### Option 1: Using Docker (Recommended)

```bash
# Start all services including the API
make up

# Or manually:
docker-compose up -d

# View logs
make logs

# Stop services
make down
```

#### Option 2: Development Mode with Hot Reloading

```bash
# Start in development mode
make up-dev

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

#### Option 3: Run API Locally

1. Start only Qdrant and Redis:
```bash
docker-compose up -d qdrant redis
```

2. Run the API:
```bash
python -m src.api.main
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage Example

```python
import requests

# Filter tools based on conversation
response = requests.post(
    "http://localhost:8000/api/v1/tools/filter",
    json={
        "messages": [
            {"role": "user", "content": "I need to search for Python files in the project"}
        ],
        "available_tools": [
            {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for patterns in files"
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "find",
                    "description": "Find files by name"
                }
            }
        ]
    }
)

print(response.json())
# {
#     "recommended_tools": [
#         {"tool_name": "find", "confidence": 0.95},
#         {"tool_name": "grep", "confidence": 0.85}
#     ],
#     "metadata": {"processing_time_ms": 42}
# }
```

## API Endpoints

### Main Endpoints

- `POST /api/v1/tools/filter` - Filter tools based on conversation context
- `GET /api/v1/tools/search` - Search tools by text query
- `POST /api/v1/tools/register` - Register new tools (for batch indexing)
- `GET /api/v1/tools/info` - Get information about indexed tools
- `GET /api/v1/collections` - List all vector store collections with metadata
- `GET /health` - Health check endpoint

### Response Format

```json
{
    "recommended_tools": [
        {
            "tool_name": "find",
            "confidence": 0.85,
            "reasoning": "High relevance to file search operations"
        }
    ],
    "metadata": {
        "processing_time_ms": 45.2,
        "embedding_model": "voyage/voyage-2",
        "total_tools_analyzed": 20,
        "conversation_messages": 3,
        "request_id": "uuid-here",
        "conversation_patterns": ["file_search", "code_analysis"]
    }
}
```

## Performance

### Current Metrics (MVP)

- **Average Latency**: ~1.7s (target: <100ms)
- **Precision@5**: 40% (target: 60%)
- **Recall@5**: 13%
- **F1 Score**: 20%

### Optimization Roadmap

1. Pre-index all tools on startup
2. Implement connection pooling
3. Add batch embedding generation
4. Optimize vector search parameters

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   FastAPI App   │────▶│  Message Parser  │
└────────┬────────┘     └──────────────────┘
         │                        │
         │              ┌─────────▼────────┐
         │              │ Embedding Service│
         │              │    (LiteLLM)     │
         │              └─────────┬────────┘
         │                        │
┌────────▼────────┐     ┌─────────▼────────┐
│  Redis Cache    │◀────│   Qdrant DB      │
└─────────────────┘     └──────────────────┘
```

## Development

### Running Tests

```bash
# Run unit tests
pytest tests/ -v

# Run evaluation
docker exec ptr_api python -m src.evaluation.run_evaluation

# Run simple API test
python test_api.py
```

### Code Quality

```bash
# Linting
ruff check src/

# Type checking
mypy src/

# Formatting
black src/
```

### Performance Testing

```bash
# Start the load test UI
locust -f tests/load_test.py
```

## Configuration

Key configuration options in `.env`:

- `PRIMARY_EMBEDDING_MODEL`: Main embedding model (default: voyage-2)
- `FALLBACK_EMBEDDING_MODEL`: Fallback model (default: text-embedding-3-small)
- `MAX_TOOLS_TO_RETURN`: Maximum tools to return (default: 10)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.7)

### Vector Store Collections

The system automatically creates model-specific collections to handle different embedding dimensions:

- Collections are named as: `tools_<model_name>` (e.g., `tools_voyage_voyage_3`)
- Each collection stores metadata including model name, dimension, and creation time
- Switching between models is seamless - the system will use the appropriate collection
- Use the `/api/v1/collections` endpoint to view all collections

**Important**: When changing embedding models, you'll need to re-index your tools as embeddings from different models are not compatible.

### Automatic Fallback Mechanism

The system supports automatic fallback to a secondary embedding model when the primary model fails:

- Configure `FALLBACK_EMBEDDING_MODEL` in your `.env` file
- Separate vector store collections are maintained for each model
- On primary model failure (e.g., rate limits, API errors), requests automatically use the fallback
- The `embedding_model` field in responses indicates which model was used
- Both models must be properly configured with valid API keys

## Documentation

See the `/documentation` directory for:
- [Technical Requirements](documentation/001-002-PTR_TRD.md)
- [Implementation TODO](documentation/001-003-PTR_TODO.md)
- [Evaluation Framework](documentation/001-006-evaluation_framework.md)

## License

[MIT License](LICENSE)