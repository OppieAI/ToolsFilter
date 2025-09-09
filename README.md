# OppieAI MCP Tool Filter

A Precision-driven Tool Recommendation ([PTR](https://arxiv.org/html/2411.09613v1)) system for filtering MCP (Model Context Protocol) tools based on conversation context. Fetch only relevant tool for the ongoing conversation and save cost while increasing the precision of your LLM Response.

Developed by [OppieAI](https://oppie.ai)

## 🎥 Explainer Video

[![OppieAI's ToolFilter](https://markdown-videos-api.jorgenkh.no/url?url=https%3A%2F%2Fyoutu.be%2FGXXbvUxqgdE)](https://youtu.be/GXXbvUxqgdE)

*Watch the full explanation of how ToolsFilter works and its impact on LLM performance*

## Table of Contents

- [🎥 Explainer Video](#-explainer-video)
- [Why?](#why)
  - [The Tool Overload Problem](#the-tool-overload-problem)
  - [Research Evidence](#-research-evidence)
  - [Visual Evidence](#-visual-evidence)
- [Solution](#solution)
  - [Core Capabilities](#core-capabilities)
  - [Infrastructure & Performance](#infrastructure--performance)
  - [Real-World Impact](#real-world-impact)
- [Architecture](#architecture)
  - [Search Pipeline Architecture](#search-pipeline-architecture)
  - [Search Strategies](#search-strategies)
- [Performance](#performance)
  - [Latest Evaluation Results](#latest-evaluation-results-august-2025)
  - [LTR Model Performance](#ltr-model-performance)
  - [Optimization Roadmap](#optimization-roadmap)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Services](#running-the-services)
  - [API Documentation](#api-documentation)
- [Usage Example](#usage-example)
- [API Endpoints](#api-endpoints)
  - [Main Endpoints](#main-endpoints)
  - [Response Format](#response-format)
- [Development](#development)
  - [Running Tests & Evaluation](#running-tests--evaluation)
  - [Latest Evaluation Reports](#latest-evaluation-reports)
  - [Code Quality](#code-quality)
  - [Performance Testing](#performance-testing)
- [Configuration](#configuration)
  - [Vector Store Collections](#vector-store-collections)
  - [Automatic Fallback Mechanism](#automatic-fallback-mechanism)
- [Documentation](#documentation)
- [References](#references)
- [License](#license)

## Why?

### The Tool Overload Problem

Modern LLMs with access to large tool suites face a critical performance degradation issue: **the more tools available, the lower the accuracy becomes**. This phenomenon is well-documented in research and practical implementations:

#### 📊 **Research Evidence**
Recent studies using [MCPGauge](https://arxiv.org/html/2508.12566v1) evaluated six commercial LLMs with 30 MCP tool suites and revealed alarming findings:

- **9.5% accuracy drop** on average when LLMs have automated access to MCP tools
- **3.25× to 236.5× increase** in input token volume, creating massive computational overhead
- **"Non-trivial friction"** between retrieved context and the model's internal reasoning
- Models struggle with instruction compliance when too many tools are available

#### 🎥 **Visual Evidence**
This accuracy degradation with increased tool count is demonstrated in [this analysis video](https://www.youtube.com/watch?v=ej7-n9OoGnQ), showing how model performance deteriorates as more tools are introduced.

## Solution

### Core Capabilities
- 🚀 **Multi-Stage Search Pipeline**: Semantic + BM25 + Cross-Encoder + LTR ranking
- 🎯 **High-Performance Results**: Perfect P@1 and MRR across all search strategies
- 🧠 **Learning-to-Rank**: XGBoost model with 46+ engineered features (NDCG@10: 0.975)
- 🔧 **OpenAI Function Calling Compatible**: Flat tool structure following OpenAI specification

### Infrastructure & Performance
- ⚡ **Multiple Embedding Providers**: Voyage AI, OpenAI, Cohere with automatic fallback
- 💾 **Intelligent Multi-Layer Caching**: Redis for queries, results, and tool indices
- 🎯 **Qdrant Vector Database**: High-performance vector search with model-specific collections
- 📊 **Comprehensive Evaluation**: Built-in framework with F1, MRR, NDCG@k metrics
- 🔄 **Message Format Compatibility**: Claude and OpenAI conversation formats
- 📝 **Collection Metadata Tracking**: Model versioning and automatic dimension handling
- 🔁 **Robust Fallback Mechanisms**: Secondary embedding models and graceful degradation

### Real-World Impact

Instead of overwhelming your LLM with 100+ tools, get precisely the 3-5 most relevant ones:
- **Before**: 236× token overhead, 9.5% accuracy loss
- **After**: 95%+ precision, perfect recall on relevant tools, minimal token usage

## Architecture

### Search Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FastAPI App   │────▶│  Message Parser  │────▶│ Search Pipeline │
└────────┬────────┘     └──────────────────┘     └─────────┬───────┘
         │                                                  │
         │                    ┌─────────────────────────────┼─────────┐
         │                    │                  │          │         │
┌────────▼────────┐    ┌──────▼─────┐  ┌─────────▼────────┐ │ ┌───────▼──────┐
│  Redis Cache    │    │ Embedding  │  │   Qdrant Vector  │ │ │ LTR Reranker │
│                 │    │ Service    │  │     Database     │ │ │  (XGBoost)   │
│ • Query Cache   │    │ (LiteLLM)  │  │                  │ │ │              │
│ • Results Cache │    │ • Voyage   │  │ • Semantic Search│ │ │ • 46 Features│
│ • Tool Index    │    │ • OpenAI   │  │ • BM25 Hybrid    │ │ │ • NDCG@10 Opt│
└─────────────────┘    │ • Fallback │  │ • Cross-Encoder  │ │ │              │
                       └────────────┘  └──────────────────┘ │ └──────────────┘
                                                            │
                                    ┌───────────────────────┘
                                    │
                             ┌──────▼──────┐
                             │ Multi-Stage │
                             │  Filtering  │
                             │             │
                             │ 1. Semantic │
                             │ 2. BM25     │
                             │ 3. Rerank   │
                             │ 4. LTR      │
                             └─────────────┘
```

### Search Strategies

1. **semantic_only**: Pure vector similarity search
2. **hybrid_basic**: BM25 + semantic search combination
3. **hybrid_cross_encoder**: + Cross-encoder reranking
4. **hybrid_ltr_full**: + Learning-to-Rank optimization

## Performance

### Latest Evaluation Results (August 2025)

**Search Strategy Comparison**: (With 300+ noise (Genuine APIs) tools to resemble real-world)

| Strategy | F1 Score | MRR | P@1 | NDCG@10 | Best For |
|----------|----------|-----|-----|---------|----------|
| **hybrid_basic** | **0.359** ⭐ | 1.000 | 1.000 | **0.975** ⭐ | General-purpose, balanced performance |
| semantic_only | 0.328 | **1.000** ⭐ | **1.000** ⭐ | 0.870 | Simple queries, exact matches |
| hybrid_cross_encoder | 0.359 | 1.000 | 1.000 | 0.964 | Complex queries requiring reranking |
| hybrid_ltr_full | 0.359 | 1.000 | 1.000 | 0.942 | Learning-based optimization |

⭐ = Best performer for that metric

📊 **[View Detailed Report](saved_eval_reports/comparison_20250823_153715.html)**

**Key Achievements**:
- **Perfect Precision@1**: All strategies achieve 1.000 P@1
- **Perfect MRR**: All strategies achieve 1.000 Mean Reciprocal Rank
- **Strong NDCG Performance**: Up to 0.975 NDCG@10 with hybrid_basic
- **Consistent F1 Scores**: 0.328-0.359 across different approaches

### LTR Model Performance

**Learning-to-Rank Training Results**:
- **Cross-Validation NDCG@10**: 0.9167 ± 0.0567
- **Training Data**: 18,354 samples with 46 features
- **Top Features**: action_alignment (32.7%), query_type_analyze (33.9%), exact_name_match (19.5%)
- **Training Speed**: <5 seconds with XGBoost

### Optimization Roadmap

✅ **Completed**:
1. ~~Pre-index all tools on startup~~ - Implemented vector store caching
2. ~~Implement connection pooling~~ - Added Redis and Qdrant connection pooling
3. ~~Add batch embedding generation~~ - Optimized embedding pipeline
4. ~~Optimize vector search parameters~~ - Tuned similarity thresholds

🎯 **In Progress**:
1. Improve LTR model with better class balancing
2. Enhance feature engineering for interaction signals
3. Optimize NDCG@5 performance for top-precision use cases

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
                "name": "grep",
                "description": "Search for patterns in files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Search pattern"}
                    },
                    "required": ["pattern"]
                },
                "strict": true
            },
            {
                "type": "function",
                "name": "find",
                "description": "Find files by name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "File name pattern"}
                    },
                    "required": ["name"]
                },
                "strict": true
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


## Development

### Running Tests & Evaluation

```bash
# Run unit tests
pytest tests/ -v

# Run comprehensive evaluation with all strategies
docker exec ptr_api python -m src.evaluation.run_evaluation

# Run strategy comparison
docker exec ptr_api python -m src.evaluation.evaluation_framework.comparison

# Train LTR model
docker exec ptr_api python -m src.scripts.train_ltr

# Run ToolBench evaluation
docker exec ptr_api python -m src.evaluation.toolbench_evaluator

# Run simple API test
python test_api.py
```

### Latest Evaluation Reports

Refer to the latest comparison report: `evaluation_results/comparison_20250823_153715.markdown`

Key findings:
- **hybrid_basic** strategy performs best overall (F1: 0.359, NDCG@10: 0.975)
- All strategies achieve perfect P@1 and MRR (1.000)
- LTR model shows consistent performance with cross-validation NDCG@10: 0.6167 ± 0.0567

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

## References
Inspired by [PTR Paper](https://arxiv.org/html/2411.09613v1)

## License

This project uses a **dual licensing model**:

- **Non-Commercial Use**: Free for research, education, and personal projects
- **Commercial Use**: Requires a separate commercial license

See [LICENSE](LICENSE) for full terms.

For commercial licensing, contact: **sales@oppie.ai**
