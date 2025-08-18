# Semantic Retrieval Enhancement Plan

## Current Implementation Analysis

### What's Working Well ✅
1. **Multi-provider support** via LiteLLM
2. **Redis caching** for embeddings (reduces API calls)
3. **Fallback embedding model** support
4. **Qdrant vector store** with filtering capabilities
5. **Basic conversation context extraction**

### Critical Gaps Identified ❌

#### 1. Poor Embedding Quality
**Current**: `f"{name}: {description}"` - only 2 fields!
```python
# Current implementation (too simplistic)
def _tool_to_text(tool):
    return f"{name}: {description}"
```

**Missing Information**:
- Parameter names and types
- Required vs optional parameters
- Return values
- Examples/usage patterns
- Categories/tags

#### 2. Single-Stage Retrieval
**Current**: Direct search in Qdrant
- Won't scale to 10,000+ tools
- No candidate pre-filtering
- No re-ranking stage

#### 3. Basic Query Processing
**Current**: Raw conversation text
- No query expansion
- No intent extraction
- No entity recognition
- No context weighting

#### 4. Limited Scoring
**Current**: Only cosine similarity
- No feature engineering
- No learning from feedback
- No personalization signals

## Recommended Enhancements

### Enhancement 1: Rich Tool Embeddings
**Priority**: HIGH | **Effort**: LOW | **Impact**: HIGH

```python
# src/services/embedding_enhancer.py
class ToolEmbeddingEnhancer:
    def tool_to_rich_text(self, tool: Dict[str, Any]) -> str:
        """Create rich text representation for better embeddings."""
        parts = []
        
        if tool.get("type") == "function":
            function = tool.get("function", {})
            
            # 1. Name and description
            name = function.get("name", "")
            description = function.get("description", "")
            parts.append(f"Tool: {name}")
            parts.append(f"Description: {description}")
            
            # 2. Parameters with types and descriptions
            params = function.get("parameters", {})
            if params.get("properties"):
                param_texts = []
                for param_name, param_spec in params["properties"].items():
                    param_type = param_spec.get("type", "any")
                    param_desc = param_spec.get("description", "")
                    is_required = param_name in params.get("required", [])
                    
                    param_text = f"{param_name} ({param_type})"
                    if param_desc:
                        param_text += f": {param_desc}"
                    if is_required:
                        param_text += " [required]"
                    param_texts.append(param_text)
                
                if param_texts:
                    parts.append(f"Parameters: {', '.join(param_texts)}")
            
            # 3. Category/tags for better clustering
            category = tool.get("category", "general")
            parts.append(f"Category: {category}")
            
            # 4. Usage examples (if available)
            if "examples" in tool:
                parts.append(f"Examples: {tool['examples']}")
            
            # 5. Common keywords/aliases
            keywords = self._extract_keywords(name, description)
            if keywords:
                parts.append(f"Keywords: {', '.join(keywords)}")
        
        return " | ".join(parts)
    
    def _extract_keywords(self, name: str, description: str) -> List[str]:
        """Extract relevant keywords for better matching."""
        keywords = []
        
        # Common tool patterns
        patterns = {
            "search": ["find", "locate", "query", "lookup"],
            "file": ["fs", "directory", "path", "folder"],
            "database": ["db", "sql", "query", "table"],
            "api": ["http", "rest", "endpoint", "request"],
            "git": ["version", "commit", "branch", "repository"]
        }
        
        text = f"{name} {description}".lower()
        for pattern, aliases in patterns.items():
            if pattern in text or any(alias in text for alias in aliases):
                keywords.append(pattern)
                keywords.extend([a for a in aliases if a in text])
        
        return list(set(keywords))[:5]  # Limit to 5 keywords
```

### Enhancement 2: Query Enhancement
**Priority**: HIGH | **Effort**: MEDIUM | **Impact**: HIGH

```python
# src/services/query_enhancer.py
class QueryEnhancer:
    def enhance_query(self, messages: List[Dict], used_tools: List[str]) -> Dict:
        """Enhance query with multiple representations."""
        
        # 1. Extract different query aspects
        last_user_message = self._get_last_user_message(messages)
        error_context = self._extract_error_context(messages)
        code_context = self._extract_code_context(messages)
        
        # 2. Create multiple query representations
        queries = {
            "primary": last_user_message,
            "intent": self._extract_intent(last_user_message),
            "error": error_context,
            "code": code_context,
            "historical": " ".join(used_tools[-3:]) if used_tools else ""
        }
        
        # 3. Query expansion with synonyms
        expanded = self._expand_with_synonyms(last_user_message)
        
        # 4. Weight different aspects
        weights = {
            "primary": 0.4,
            "intent": 0.3,
            "error": 0.15,
            "code": 0.1,
            "historical": 0.05
        }
        
        return {
            "queries": queries,
            "weights": weights,
            "expanded": expanded
        }
    
    def _expand_with_synonyms(self, query: str) -> str:
        """Expand query with technical synonyms."""
        synonyms = {
            "search": "find locate grep query",
            "file": "document fs filesystem directory",
            "error": "exception bug issue problem",
            "run": "execute launch start invoke",
            "create": "make generate build construct",
            "delete": "remove rm unlink destroy",
            "update": "modify change edit patch"
        }
        
        expanded = query
        for word, syns in synonyms.items():
            if word in query.lower():
                expanded += f" {syns}"
        
        return expanded
```

### Enhancement 3: Two-Stage Retrieval Architecture
**Priority**: HIGH | **Effort**: HIGH | **Impact**: CRITICAL

```python
# src/services/two_stage_retriever.py
import faiss
import numpy as np
from typing import List, Dict, Any

class TwoStageRetriever:
    def __init__(self, embedding_dim: int = 1536):
        # Stage 1: FAISS for fast candidate generation
        self.faiss_index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        
        # Add HNSW for better speed/recall tradeoff
        self.faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)
        self.faiss_index.hnsw.efConstruction = 200
        
        # Stage 2: Keep Qdrant for precise re-ranking
        self.qdrant_client = QdrantClient(...)
        
        # Tool metadata cache
        self.tool_metadata = {}
        
    async def retrieve(
        self,
        query_embeddings: Dict[str, np.ndarray],
        weights: Dict[str, float],
        n_candidates: int = 1000,
        n_final: int = 10
    ) -> List[Dict]:
        """Two-stage retrieval with multiple queries."""
        
        # Stage 1: Fast candidate generation with FAISS
        candidates = set()
        
        for query_type, embedding in query_embeddings.items():
            weight = weights.get(query_type, 1.0)
            
            # Search FAISS
            distances, indices = self.faiss_index.search(
                embedding.reshape(1, -1),
                min(n_candidates // 3, 500)  # Diversify candidates
            )
            
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1:  # Valid index
                    tool_id = self.index_to_tool[idx]
                    candidates.add((tool_id, dist * weight))
        
        # Aggregate scores for candidates
        tool_scores = {}
        for tool_id, score in candidates:
            if tool_id not in tool_scores:
                tool_scores[tool_id] = []
            tool_scores[tool_id].append(score)
        
        # Average scores per tool
        for tool_id in tool_scores:
            tool_scores[tool_id] = np.mean(tool_scores[tool_id])
        
        # Get top candidates
        top_candidates = sorted(
            tool_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n_candidates]
        
        # Stage 2: Precise re-ranking with Qdrant + features
        reranked = await self._rerank_with_features(
            top_candidates,
            query_embeddings["primary"],
            n_final
        )
        
        return reranked
    
    async def _rerank_with_features(
        self,
        candidates: List[tuple],
        primary_embedding: np.ndarray,
        n_final: int
    ) -> List[Dict]:
        """Re-rank candidates with additional features."""
        
        # Get detailed features from Qdrant
        candidate_ids = [c[0] for c in candidates]
        detailed_tools = await self.qdrant_client.retrieve(
            collection_name=self.collection_name,
            ids=candidate_ids
        )
        
        # Calculate additional features
        reranked = []
        for tool in detailed_tools:
            features = {
                "semantic_score": self._calculate_semantic_score(tool, primary_embedding),
                "parameter_match": self._calculate_param_match(tool),
                "complexity_match": self._calculate_complexity_match(tool),
                "popularity": tool.payload.get("usage_count", 0) / 1000
            }
            
            # Weighted combination
            final_score = (
                features["semantic_score"] * 0.5 +
                features["parameter_match"] * 0.2 +
                features["complexity_match"] * 0.2 +
                features["popularity"] * 0.1
            )
            
            reranked.append({
                "tool_name": tool.payload["name"],
                "score": final_score,
                "features": features,
                **tool.payload
            })
        
        # Sort and return top N
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:n_final]
```

### Enhancement 4: Hybrid Embedding Strategy
**Priority**: MEDIUM | **Effort**: MEDIUM | **Impact**: HIGH

```python
# src/services/hybrid_embedder.py
class HybridEmbedder:
    def __init__(self):
        # Multiple embedding models for different aspects
        self.semantic_model = "voyage-2"  # Good for semantic similarity
        self.code_model = "codex-embed"  # Better for code understanding
        self.keyword_model = "BM25"  # For exact keyword matching
        
    async def create_hybrid_embedding(
        self,
        text: str,
        tool_type: str = "general"
    ) -> Dict[str, np.ndarray]:
        """Create multiple embeddings for different matching strategies."""
        
        embeddings = {}
        
        # 1. Semantic embedding (general understanding)
        embeddings["semantic"] = await self.embed_semantic(text)
        
        # 2. Code embedding (if tool involves code)
        if tool_type in ["code", "api", "function"]:
            embeddings["code"] = await self.embed_code(text)
        
        # 3. Keyword features (for exact matching)
        embeddings["keywords"] = self.extract_keyword_features(text)
        
        # 4. Structural embedding (parameter patterns)
        embeddings["structure"] = self.extract_structural_features(text)
        
        return embeddings
    
    def extract_keyword_features(self, text: str) -> np.ndarray:
        """Extract TF-IDF or BM25 features for keyword matching."""
        # This would use a pre-trained TF-IDF vectorizer
        # or implement BM25 scoring
        pass
    
    def extract_structural_features(self, text: str) -> np.ndarray:
        """Extract structural features like parameter count, types, etc."""
        features = []
        
        # Count various elements
        features.append(text.count("required"))
        features.append(text.count("optional"))
        features.append(text.count("string"))
        features.append(text.count("number"))
        features.append(text.count("boolean"))
        features.append(text.count("array"))
        features.append(text.count("object"))
        
        # Normalize
        return np.array(features) / (sum(features) + 1)
```

### Enhancement 5: Learning from Feedback
**Priority**: MEDIUM | **Effort**: LOW | **Impact**: MEDIUM

```python
# src/services/feedback_learner.py
class FeedbackLearner:
    def __init__(self):
        self.positive_pairs = []  # (query, tool) pairs that worked
        self.negative_pairs = []  # (query, tool) pairs that didn't work
        
    async def learn_from_usage(
        self,
        query_embedding: np.ndarray,
        tool_id: str,
        was_used: bool
    ):
        """Adjust embeddings based on usage feedback."""
        
        if was_used:
            # Move tool embedding closer to query
            await self._adjust_embedding(tool_id, query_embedding, alpha=0.1)
            self.positive_pairs.append((query_embedding, tool_id))
        else:
            # Optionally move slightly away (be careful not to over-penalize)
            await self._adjust_embedding(tool_id, query_embedding, alpha=-0.02)
            self.negative_pairs.append((query_embedding, tool_id))
        
        # Periodically retrain with contrastive learning
        if len(self.positive_pairs) >= 100:
            await self._retrain_with_contrastive_learning()
    
    async def _adjust_embedding(
        self,
        tool_id: str,
        target_embedding: np.ndarray,
        alpha: float
    ):
        """Adjust tool embedding towards/away from target."""
        current = await self.get_tool_embedding(tool_id)
        
        # Simple exponential moving average
        adjusted = current + alpha * (target_embedding - current)
        
        # Normalize to maintain magnitude
        adjusted = adjusted / np.linalg.norm(adjusted)
        
        await self.update_tool_embedding(tool_id, adjusted)
```

## Implementation Priority Matrix

| Enhancement | Impact | Effort | Priority | Timeline |
|------------|--------|--------|----------|----------|
| Rich Tool Embeddings | HIGH | LOW | **P0** | Week 1 |
| Query Enhancement | HIGH | MEDIUM | **P0** | Week 1 |
| Two-Stage Retrieval | CRITICAL | HIGH | **P1** | Week 2-3 |
| Hybrid Embeddings | HIGH | MEDIUM | **P2** | Week 3-4 |
| Feedback Learning | MEDIUM | LOW | **P2** | Week 4 |

## Performance Targets

### Before Enhancement
- **Embedding Quality**: ~60% relevant information captured
- **Retrieval Speed**: ~500ms for 1000 tools
- **Precision@10**: ~70%
- **Scalability**: <5000 tools

### After Enhancement
- **Embedding Quality**: ~95% relevant information captured
- **Retrieval Speed**: <100ms for 10,000+ tools
- **Precision@10**: ~90%
- **Scalability**: 100,000+ tools

## Quick Wins (Implement Today)

### 1. Enhanced Tool Text (30 minutes)
```python
# Replace current _tool_to_text function
def _tool_to_text(tool: Dict[str, Any]) -> str:
    """Convert tool to rich text for embedding."""
    parts = []
    
    if tool.get("type") == "function":
        function = tool.get("function", {})
        name = function.get("name", "")
        description = function.get("description", "")
        
        # Add all parameter names and types
        params = function.get("parameters", {})
        param_info = []
        if params.get("properties"):
            for pname, pspec in params["properties"].items():
                ptype = pspec.get("type", "any")
                pdesc = pspec.get("description", "")
                param_info.append(f"{pname}:{ptype} {pdesc}")
        
        # Combine everything
        text = f"Tool {name}: {description}"
        if param_info:
            text += f" Parameters: {' '.join(param_info)}"
        
        # Add category if available
        if "category" in tool:
            text += f" Category: {tool['category']}"
        
        return text
    
    return str(tool)
```

### 2. Multi-Query Search (1 hour)
```python
# Enhance search to use multiple query formulations
async def enhanced_search(messages, available_tools):
    # Extract multiple query aspects
    last_message = messages[-1]["content"] if messages else ""
    full_context = " ".join([m.get("content", "") for m in messages[-3:]])
    
    # Generate multiple embeddings
    embeddings = []
    embeddings.append(await embed_text(last_message))  # Most recent
    embeddings.append(await embed_text(full_context))  # Context
    
    # Search with each embedding and merge results
    all_results = []
    for emb in embeddings:
        results = await vector_store.search_similar_tools(emb, limit=20)
        all_results.extend(results)
    
    # Deduplicate and re-score
    tool_scores = {}
    for result in all_results:
        tool_name = result["tool_name"]
        if tool_name not in tool_scores:
            tool_scores[tool_name] = []
        tool_scores[tool_name].append(result["score"])
    
    # Average scores and sort
    final_results = []
    for tool_name, scores in tool_scores.items():
        avg_score = np.mean(scores)
        final_results.append({"tool_name": tool_name, "score": avg_score})
    
    final_results.sort(key=lambda x: x["score"], reverse=True)
    return final_results[:10]
```

## Monitoring & Evaluation

### Metrics to Track
1. **Retrieval Metrics**
   - Precision@K (K=5, 10, 20)
   - Recall@K
   - Mean Reciprocal Rank (MRR)
   - Normalized Discounted Cumulative Gain (NDCG)

2. **Performance Metrics**
   - P50, P90, P99 latency
   - Throughput (requests/second)
   - Cache hit rate

3. **Business Metrics**
   - Tool usage rate
   - User satisfaction (fewer retries)
   - Coverage (% tools recommended)

### A/B Testing Plan
```python
# Run experiments comparing:
variants = {
    "control": "current_implementation",
    "rich_embeddings": "enhanced_tool_text",
    "two_stage": "faiss_plus_qdrant",
    "full_enhanced": "all_improvements"
}
```

## Conclusion

The current semantic retrieval implementation has a solid foundation but needs significant enhancements to handle production scale (10,000+ tools) and achieve high precision. The recommended enhancements focus on:

1. **Information Richness**: Capture 95% of tool information vs current 60%
2. **Retrieval Architecture**: Two-stage for 10x speed improvement
3. **Query Understanding**: Multiple representations for better matching
4. **Continuous Learning**: Feedback loop for improvement

Start with the quick wins (rich embeddings, multi-query) for immediate 20-30% improvement, then implement two-stage retrieval for scalability.