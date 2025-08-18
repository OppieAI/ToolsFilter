# Conversion Plan: From Semantic Search to Hybrid Contextual Bandit System

## Executive Summary
Transform the current semantic-search-only implementation into a production-ready hybrid system combining contextual bandits (LinUCB) with semantic search, supporting 10,000+ tools with <100ms latency.

## Current State Analysis

### Existing Components (Reusable)
1. **Embedding Infrastructure** ✅
   - LiteLLM for embeddings (supports multiple providers)
   - Redis caching layer for embeddings
   - Batch processing capability

2. **Vector Store** ✅
   - Qdrant for similarity search
   - Collection management with metadata
   - Filtering capabilities

3. **API Framework** ✅
   - FastAPI endpoints
   - Request/response models
   - Authentication system

4. **Message Parsing** ✅
   - Conversation context extraction
   - User intent detection
   - Pattern analysis

### Critical Gaps to Address
1. **No Learning Mechanism** ❌
   - No usage tracking
   - No feedback loop
   - No online learning

2. **No Exploration Strategy** ❌
   - No handling of new tools
   - No UCB/Thompson Sampling
   - Risk of stagnation

3. **No Personalization** ❌
   - User ID not utilized
   - No per-user statistics
   - No contextual features

4. **Limited Scalability** ❌
   - No two-stage retrieval
   - No ANN optimization
   - Single-model dependency

## Phase 0: Semantic Retrieval Enhancement (Week 1)
**Goal**: Improve existing semantic search quality (See 003-003-semantic-retrieval-enhancements.md)

### Quick Wins (Immediate)
- Rich tool embeddings (capture parameters, types, categories)
- Multi-query search (last message + context)
- Query expansion with synonyms

### Critical Improvements
- Two-stage retrieval architecture (FAISS + Qdrant)
- Hybrid embedding strategy
- Better text representation

## Phase 1: Foundation (Week 1-2)
**Goal**: Add usage tracking and feedback mechanism

### 1.1 Database Schema Extension
```python
# New models in src/core/models.py
class ToolUsageEvent(BaseModel):
    tool_name: str
    user_id: Optional[str]
    conversation_id: str
    timestamp: datetime
    was_used: bool
    context_embedding: Optional[List[float]]
    
class ToolStatistics(BaseModel):
    tool_name: str
    total_recommendations: int = 0
    total_uses: int = 0
    last_used: Optional[datetime]
    mean_reward: float = 0.0
    confidence_bound: float = 1.0
```

### 1.2 Add Feedback Endpoint
```python
# New endpoint in src/api/endpoints.py
@router.post("/tools/feedback")
async def record_tool_usage(
    conversation_id: str,
    tool_name: str,
    was_used: bool,
    user_id: Optional[str] = None
):
    # Record usage in PostgreSQL/Redis
    # Update tool statistics
```

### 1.3 PostgreSQL Integration
**Library**: `asyncpg==0.29.0` + `sqlalchemy[asyncio]==2.0.23`

```python
# src/services/usage_store.py
class UsageStore:
    async def record_usage(self, event: ToolUsageEvent)
    async def get_tool_stats(self, tool_name: str) -> ToolStatistics
    async def update_statistics(self, tool_name: str, reward: float)
```

### Libraries to Add:
```txt
# Database
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.1  # For migrations

# Time series (optional for analytics)
influxdb-client==1.38.0
```

## Phase 2: Contextual Bandit Integration (Week 3-4)
**Goal**: Implement Thompson Sampling with exploration using production-ready libraries

### 2.1 Thompson Sampling Implementation
**Library**: `pybandits==0.6.0` (PlaytikaOSS) or `contextualbandits==0.3.17` (david-cortes)

#### Option A: PyBandits (Recommended for Bayesian approach)
```python
# Installation
# conda install -c conda-forge pymc
# pip install pybandits

# src/services/bandit.py
from pybandits.model import Beta, Gaussian
from pybandits.cmab import CmabBernoulli, CmabGaussian
import numpy as np
from typing import Dict, List, Optional

class ThompsonSamplingBandit:
    def __init__(self, n_features: int, n_tools: int):
        """
        Initialize Thompson Sampling for contextual bandits
        n_features: dimension of context features
        n_tools: number of tools (arms)
        """
        # For binary rewards (tool used/not used)
        self.actions = {
            f"tool_{i}": Beta() for i in range(n_tools)
        }
        self.cmab = CmabBernoulli(actions=self.actions)
        
        # Store tool mapping
        self.tool_id_to_index = {}
        self.index_to_tool_id = {}
        
    def predict(self, context_features: np.ndarray, available_tools: List[str], n_recommendations: int = 10):
        """Select tools using Thompson Sampling"""
        # Map tool IDs to indices
        for idx, tool_id in enumerate(available_tools):
            if tool_id not in self.tool_id_to_index:
                self.tool_id_to_index[tool_id] = len(self.tool_id_to_index)
                self.index_to_tool_id[self.tool_id_to_index[tool_id]] = tool_id
        
        # Get predictions from bandit
        pred_actions, scores = self.cmab.predict(
            n_samples=n_recommendations,
            context=context_features
        )
        
        return pred_actions, scores
    
    def update(self, tool_ids: List[str], rewards: List[float], contexts: np.ndarray):
        """Update posterior distributions based on observed rewards"""
        actions = [f"tool_{self.tool_id_to_index[tid]}" for tid in tool_ids]
        self.cmab.update(actions=actions, rewards=rewards, context=contexts)
```

#### Option B: contextualbandits (More flexible, works with any sklearn classifier)
```python
# Installation
# pip install contextualbandits

# src/services/bandit.py
from contextualbandits.online import BootstrappedTS
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Dict, List, Optional

class ThompsonSamplingBandit:
    def __init__(self, n_features: int, n_tools: int, n_bootstraps: int = 10):
        """
        Initialize Thompson Sampling using bootstrap
        n_features: dimension of context features
        n_tools: number of tools (arms)
        n_bootstraps: number of bootstrap samples for Thompson Sampling
        """
        # Use logistic regression as base model
        base_model = LogisticRegression(solver='lbfgs', warm_start=True)
        
        # Initialize Thompson Sampling with bootstrapping
        self.bandit = BootstrappedTS(
            base_algorithm=base_model,
            nchoices=n_tools,
            nsamples=n_bootstraps,
            njobs=-1  # Use all CPU cores
        )
        
        # Store tool mapping
        self.tool_id_to_index = {}
        self.index_to_tool_id = {}
        self.n_tools = n_tools
        
    def predict(self, context_features: np.ndarray, available_tools: List[str]) -> List[tuple]:
        """Select tools using Thompson Sampling"""
        # Map tool IDs to indices
        for tool_id in available_tools:
            if tool_id not in self.tool_id_to_index:
                idx = len(self.tool_id_to_index)
                self.tool_id_to_index[tool_id] = idx
                self.index_to_tool_id[idx] = tool_id
        
        # Get predictions (returns indices of selected arms)
        selected_arms = self.bandit.predict(context_features)
        
        # Convert indices back to tool IDs with scores
        results = []
        for arm_idx in selected_arms:
            if arm_idx in self.index_to_tool_id:
                tool_id = self.index_to_tool_id[arm_idx]
                # Get probability scores for ranking
                probs = self.bandit.predict_proba(context_features)
                score = probs[0][arm_idx] if arm_idx < len(probs[0]) else 0.0
                results.append((tool_id, score))
        
        return results
    
    def update(self, context: np.ndarray, tool_id: str, reward: float):
        """Update model based on observed reward"""
        if tool_id in self.tool_id_to_index:
            arm_idx = self.tool_id_to_index[tool_id]
            # Partial fit with new observation
            self.bandit.partial_fit(
                X=context.reshape(1, -1),
                a=np.array([arm_idx]),
                r=np.array([reward])
            )
```

### 2.1.1 Comparison: PyBandits vs contextualbandits

| Feature | PyBandits | contextualbandits |
|---------|-----------|-------------------|
| **Approach** | Bayesian (PyMC) | Bootstrap + Any Classifier |
| **Flexibility** | Medium | High (any sklearn model) |
| **Performance** | Good for small-medium scale | Better for large scale |
| **Learning** | Full Bayesian | Frequentist with bootstrap |
| **Production Ready** | Yes (Playtika uses it) | Yes (more mature) |
| **Parallelization** | Limited | Full (with njobs) |

**Recommendation**: Use `contextualbandits` for better scalability and flexibility

### 2.2 Feature Extraction
```python
# src/services/feature_extractor.py
class FeatureExtractor:
    def extract_features(
        self,
        conversation_embedding: np.ndarray,
        tool_embedding: np.ndarray,
        tool_stats: ToolStatistics,
        conversation_metadata: Dict
    ) -> np.ndarray:
        """Extract features for contextual bandit"""
        features = []
        
        # Semantic similarity
        cosine_sim = np.dot(conversation_embedding, tool_embedding)
        features.append(cosine_sim)
        
        # Usage statistics (with time decay)
        time_since_last_use = (datetime.now() - tool_stats.last_used).seconds if tool_stats.last_used else 1e6
        features.append(np.exp(-0.01 * time_since_last_use))
        
        # Popularity (normalized)
        features.append(tool_stats.mean_reward)
        
        # Conversation features
        features.append(float(conversation_metadata.get("has_error", False)))
        features.append(float(conversation_metadata.get("has_code", False)))
        features.append(len(conversation_metadata.get("messages", [])) / 100)
        
        return np.array(features)
```

### Libraries to Add:
```txt
# Scientific computing (already in requirements)
numpy==1.26.4  # Already present
scipy==1.11.4  # For advanced stats
```

## Phase 3: Two-Stage Retrieval (Week 5-6)
**Goal**: Scale to 10,000+ tools with FAISS

### 3.1 FAISS Integration
**Library**: `faiss-cpu==1.7.4` or `faiss-gpu==1.7.4`

```python
# src/services/ann_index.py
import faiss
import numpy as np

class ANNIndex:
    def __init__(self, dimension: int, use_gpu: bool = False):
        # Use HNSW for best recall/speed tradeoff
        self.index = faiss.IndexHNSWFlat(dimension, 32)
        self.index.hnsw.efConstruction = 200
        
        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )
        
        self.tool_ids = []
        
    def add_tools(self, embeddings: np.ndarray, tool_ids: List[str]):
        """Add tool embeddings to index"""
        self.index.add(embeddings)
        self.tool_ids.extend(tool_ids)
        
    def search(self, query: np.ndarray, k: int = 1000) -> List[tuple]:
        """Fast approximate nearest neighbor search"""
        distances, indices = self.index.search(query.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.tool_ids):
                results.append((self.tool_ids[idx], float(dist)))
        
        return results
```

### 3.2 Hybrid Ranker
**Library**: `xgboost==2.0.3` for Learning-to-Rank

```python
# src/services/ranker.py
import xgboost as xgb

class HybridRanker:
    def __init__(self):
        self.semantic_weight = 0.6
        self.bandit_weight = 0.3
        self.exploration_weight = 0.1
        
        # Optional: XGBoost ranker for learned combination
        self.ltr_model = None
        
    def rank_tools(
        self,
        candidates: List[str],
        semantic_scores: Dict[str, float],
        bandit_scores: Dict[str, float],
        exploration_scores: Dict[str, float],
        features: Optional[np.ndarray] = None
    ) -> List[tuple]:
        """Combine scores from different sources"""
        
        if self.ltr_model and features is not None:
            # Use learned ranking model
            scores = self.ltr_model.predict(features)
        else:
            # Use weighted combination
            scores = {}
            for tool_id in candidates:
                score = (
                    self.semantic_weight * semantic_scores.get(tool_id, 0) +
                    self.bandit_weight * bandit_scores.get(tool_id, 0) +
                    self.exploration_weight * exploration_scores.get(tool_id, 0)
                )
                scores[tool_id] = score
        
        # Sort by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked
```

### Libraries to Add:
```txt
# ANN Search
faiss-cpu==1.7.4  # Or faiss-gpu for GPU support

# Learning to Rank
xgboost==2.0.3
lightgbm==4.1.0  # Alternative to XGBoost

# Feature engineering
scikit-learn==1.3.2
```

## Phase 4: Advanced Features (Week 7-8)
**Goal**: Add monitoring, A/B testing, and optimization

### 4.1 Experiment Tracking
**Library**: `mlflow==2.9.2` (already in requirements)

```python
# src/services/experiment_tracker.py
import mlflow
from mlflow.tracking import MlflowClient

class ExperimentTracker:
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        
    def log_recommendation(
        self,
        conversation_id: str,
        recommended_tools: List[str],
        algorithm_version: str,
        metrics: Dict
    ):
        with mlflow.start_run(run_name=f"rec_{conversation_id}"):
            mlflow.log_param("algorithm", algorithm_version)
            mlflow.log_param("num_tools", len(recommended_tools))
            mlflow.log_metrics(metrics)
```

### 4.2 A/B Testing Framework
**Library**: `statsmodels==0.14.1`

```python
# src/services/ab_testing.py
class ABTestManager:
    def __init__(self):
        self.experiments = {}
        
    def assign_variant(self, user_id: str, experiment_name: str) -> str:
        """Assign user to experiment variant"""
        # Use consistent hashing for assignment
        hash_val = hashlib.md5(f"{user_id}:{experiment_name}".encode()).hexdigest()
        return "treatment" if int(hash_val, 16) % 2 == 0 else "control"
    
    def should_use_bandit(self, user_id: str) -> bool:
        """Determine if user should use bandit algorithm"""
        variant = self.assign_variant(user_id, "bandit_vs_semantic")
        return variant == "treatment"
```

### 4.3 Performance Monitoring
**Library**: `prometheus-client==0.19.0`

```python
# src/services/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Metrics
recommendation_latency = Histogram(
    'tool_recommendation_latency_seconds',
    'Time to generate recommendations'
)

tool_usage_rate = Counter(
    'tool_usage_total',
    'Number of tool uses',
    ['tool_name', 'was_used']
)

active_tools = Gauge(
    'active_tools_count',
    'Number of active tools in system'
)
```

### Libraries to Add:
```txt
# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Statistical analysis
statsmodels==0.14.1
pandas==2.1.4  # For data analysis
```

## Phase 5: Production Optimization (Week 9-10)
**Goal**: Optimize for production scale

### 5.1 Caching Strategy
```python
# Enhance Redis caching
# src/services/cache_manager.py
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(...)
        
    async def get_or_compute(
        self,
        key: str,
        compute_fn: callable,
        ttl: int = 3600
    ):
        """Get from cache or compute and cache"""
        cached = self.redis_client.get(key)
        if cached:
            return json.loads(cached)
        
        result = await compute_fn()
        self.redis_client.setex(key, ttl, json.dumps(result))
        return result
```

### 5.2 Batch Inference Optimization
```python
# src/services/batch_processor.py
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_batch(
        self,
        items: List[Any],
        process_fn: callable,
        batch_size: int = 100
    ):
        """Process items in parallel batches"""
        tasks = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            task = asyncio.create_task(
                asyncio.to_thread(process_fn, batch)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
```

## Migration Strategy

### Step 1: Non-Breaking Additions (Week 1-2)
1. Add usage tracking tables (no impact on existing API)
2. Add feedback endpoint (new endpoint)
3. Start collecting data in shadow mode

### Step 2: Parallel Implementation (Week 3-6)
1. Implement LinUCB in parallel with existing system
2. Add feature extraction pipeline
3. Integrate FAISS for candidate generation
4. Run both systems, log metrics for comparison

### Step 3: Gradual Rollout (Week 7-8)
1. A/B test with 10% traffic
2. Monitor metrics (CTR, latency, coverage)
3. Gradually increase traffic to new system

### Step 4: Full Migration (Week 9-10)
1. Switch all traffic to new system
2. Deprecate old endpoints
3. Optimize based on production metrics

## Updated Dependencies

```txt
# Add to requirements.txt

# Contextual Bandits (choose one)
contextualbandits==0.3.17  # Recommended: more mature, better scalability
# OR
pybandits==0.6.0  # Alternative: if preferring Bayesian approach
# pymc==5.10.0  # Required for pybandits (install via conda first)

# Core ML/Stats
scipy==1.11.4
scikit-learn==1.3.2
xgboost==2.0.3

# Database
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.1

# Vector Search
faiss-cpu==1.7.4  # or faiss-gpu==1.7.4

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.21.0
opentelemetry-sdk==1.21.0

# Analysis
statsmodels==0.14.1
pandas==2.1.4
```

## Success Metrics

### Primary Metrics
1. **Tool Usage Rate**: % of recommended tools that get used
2. **Coverage**: % of available tools recommended at least once per week
3. **Latency P99**: <100ms for 10,000 tools
4. **User Satisfaction**: Reduction in "tool not found" errors

### Secondary Metrics
1. **Exploration Rate**: % of new tools in recommendations
2. **Personalization Impact**: Lift in usage rate for personalized recommendations
3. **System Reliability**: 99.9% uptime
4. **Cache Hit Rate**: >80% for popular queries

## Risk Mitigation

### Technical Risks
1. **Thompson Sampling Cold Start**: Use Beta(1,1) or Gaussian priors + content-based features
2. **FAISS Memory Usage**: Use IVF index for very large scale
3. **Latency Spikes**: Implement circuit breakers and timeouts
4. **Data Loss**: Regular backups of usage statistics

### Business Risks
1. **Poor Initial Performance**: Keep fallback to pure semantic search
2. **User Confusion**: Gradual rollout with monitoring
3. **Increased Complexity**: Comprehensive monitoring and alerting

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Semantic Enhancement | Rich embeddings, multi-query, query expansion |
| 2-3 | Foundation | Usage tracking, feedback loop |
| 4-5 | Bandit Integration | Thompson Sampling implementation, feature extraction |
| 6-7 | Two-Stage Retrieval | FAISS integration, hybrid ranking |
| 8-9 | Advanced Features | A/B testing, monitoring |
| 10 | Production Optimization | Performance tuning, full rollout |

## Next Steps

1. **Immediate Actions**:
   - Set up PostgreSQL for usage tracking
   - Create database migration scripts
   - Implement feedback endpoint

2. **Design Reviews**:
   - Review Thompson Sampling parameters with ML team
   - Validate feature engineering approach
   - Plan A/B testing strategy

3. **Infrastructure**:
   - Provision PostgreSQL instance
   - Set up MLflow tracking server
   - Configure Prometheus monitoring

This plan transforms the current semantic-only system into a state-of-the-art hybrid recommendation system, following industry best practices from companies like Microsoft, Google, and Spotify.