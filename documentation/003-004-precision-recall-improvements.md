# Precision-Recall Improvements for PTR Tool Filter

## Current Performance Baseline
- **Precision**: 59.3%
- **Recall**: 80.5%
- **F1 Score**: 63.0%
- **Model**: OpenAI text-embedding-3-small
- **Threshold**: 0.352 (optimized)

## Implementation Plan: Three High-Impact Improvements

### 1. BM25 Hybrid Search (Lexical + Semantic)

#### Rationale
- Semantic embeddings miss exact keyword matches
- Users often mention specific tool names or technical terms
- BM25 excels at exact and partial keyword matching
- Hybrid approach proven in production systems (Elasticsearch, Google)

#### Implementation Details

**Components:**
- `src/services/bm25_ranker.py` - BM25 ranking service
- Integration in `vector_store.py` for hybrid scoring
- Tokenization optimized for tool names and technical terms

**Architecture:**
```python
class BM25Ranker:
    def __init__(self):
        self.tokenizer = self._create_technical_tokenizer()
        self.bm25 = None
        self.tool_corpus = []
        
    def index_tools(self, tools):
        # Create searchable corpus from tool names, descriptions, parameters
        self.tool_corpus = [self._tool_to_text(tool) for tool in tools]
        tokenized = [self.tokenizer(text) for text in self.tool_corpus]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, k=100):
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        return scores
```

**Hybrid Scoring Formula:**
```
final_score = α * semantic_score + β * bm25_score
where α = 0.7, β = 0.3 (tunable)
```

**Expected Impact:**
- Recall: +5-8% (better exact matches)
- Precision: +2-3% (reduced false positives)

### 2. Cross-Encoder Reranking

#### Rationale
- Bi-encoders (current approach) encode query and tools separately
- Cross-encoders process query-tool pairs jointly for higher accuracy
- Only applied to top-100 candidates (computationally feasible)
- Microsoft's ms-marco models proven effective for tool/API ranking

#### Implementation Details

**Components:**
- `src/services/cross_encoder_reranker.py` - Reranking service
- Integration as Stage 3 in retrieval pipeline
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, accurate)

**Architecture:**
```python
class CrossEncoderReranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model = CrossEncoder(model_name, max_length=512)
        self.cache = LRUCache(maxsize=1000)
        
    async def rerank(self, query, candidates, top_k=10):
        # Create query-tool pairs
        pairs = [(query, self._tool_to_text(tool)) for tool in candidates]
        
        # Get cross-encoder scores (batch processing)
        ce_scores = self.model.predict(pairs, batch_size=32)
        
        # Combine with original scores
        combined = self._combine_scores(candidates, ce_scores)
        
        return sorted(combined, key=lambda x: x['score'], reverse=True)[:top_k]
```

**Three-Stage Pipeline:**
1. Stage 1: Semantic + BM25 retrieval (top-100)
2. Stage 2: Cross-encoder reranking (top-100 → top-30)
3. Stage 3: Final scoring with all signals (top-10)

**Expected Impact:**
- Precision: +10-15% (much better relevance assessment)
- Minimal impact on recall (reranking existing candidates)

### 3. Learning to Rank (LTR) with XGBoost

#### Rationale
- Combines multiple signals beyond similarity scores
- Learns optimal feature weights from evaluation data
- Can incorporate usage patterns and contextual features
- Industry standard (Airbnb, Uber, LinkedIn)

#### Implementation Details

**Components:**
- `src/services/ltr_ranker.py` - Learning to Rank service
- Feature extraction pipeline
- Training pipeline using evaluation data
- Online feature computation for inference

**Feature Set:**
```python
class LTRFeatureExtractor:
    def extract_features(self, query, tool, context):
        return {
            # Similarity features
            'semantic_similarity': cosine_similarity,
            'bm25_score': bm25_score,
            'cross_encoder_score': ce_score,
            
            # Name matching features
            'exact_name_match': query_has_tool_name,
            'partial_name_match': partial_name_overlap,
            'edit_distance': levenshtein_distance,
            
            # Description features
            'description_length': len(tool.description),
            'description_overlap': jaccard_similarity,
            'keyword_density': keyword_matches / description_length,
            
            # Parameter features
            'num_required_params': len(required_params),
            'num_optional_params': len(optional_params),
            'param_complexity': calculate_param_complexity(),
            
            # Query features
            'query_length': len(query.split()),
            'query_type': classify_query_type(query),
            'has_code_snippet': detect_code_in_query(query),
            
            # Tool metadata
            'tool_category': tool.category,
            'tool_popularity': historical_usage_count,
            'tool_recency': days_since_last_use,
        }
```

**Training Pipeline:**
```python
class LTRTrainer:
    def train_from_evaluations(self, evaluation_results):
        # Convert evaluation data to training format
        X, y = self._prepare_training_data(evaluation_results)
        
        # Train XGBoost ranker
        self.model = xgb.XGBRanker(
            objective='rank:pairwise',
            learning_rate=0.1,
            max_depth=6,
            n_estimators=100
        )
        
        self.model.fit(X, y, group=query_groups)
        
        # Feature importance analysis
        self.feature_importance = self.model.get_score(importance_type='gain')
```

**Integration:**
```python
# Final scoring in vector_store.py
def compute_final_scores(self, query, candidates):
    # Extract all features
    features = self.ltr_extractor.extract_features_batch(query, candidates)
    
    # Get LTR predictions
    ltr_scores = self.ltr_model.predict(features)
    
    # Calibrate and return
    return self._calibrate_scores(ltr_scores)
```

**Expected Impact:**
- Precision: +8-12% (optimal feature weighting)
- Recall: +3-5% (better understanding of relevance)
- F1 Score: +10-15% overall improvement

## Implementation Strategy

### Phase 1: BM25 Hybrid Search (Day 1-2)
1. Implement BM25Ranker class
2. Add tokenizer for technical terms
3. Integrate with vector_store.py
4. Test on evaluation dataset

### Phase 2: Cross-Encoder Reranking (Day 3-4)
1. Implement CrossEncoderReranker class
2. Add caching for efficiency
3. Integrate as Stage 2 in pipeline
4. Benchmark latency impact

### Phase 3: Learning to Rank (Day 5-7)
1. Implement feature extractor
2. Create training pipeline
3. Train on evaluation data
4. Deploy and validate

### Phase 4: Optimization & Testing (Day 8-9)
1. Hyperparameter tuning
2. A/B testing setup
3. Performance benchmarking
4. Documentation updates

## Configuration

### New Settings (config.py)
```python
# BM25 Configuration
bm25_enabled: bool = True
bm25_weight: float = 0.3
bm25_k1: float = 1.2  # Term frequency saturation
bm25_b: float = 0.75  # Length normalization

# Cross-Encoder Configuration
cross_encoder_enabled: bool = True
cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
cross_encoder_batch_size: int = 32
cross_encoder_cache_size: int = 1000

# LTR Configuration
ltr_enabled: bool = True
ltr_model_path: str = "./models/ltr_xgboost.json"
ltr_feature_version: str = "v1"
ltr_retrain_threshold: int = 1000  # Retrain after N new evaluations
```

## Expected Performance Gains

### Conservative Estimates
- **Precision**: 59.3% → 72% (+12.7%)
- **Recall**: 80.5% → 86% (+5.5%)
- **F1 Score**: 63.0% → 78% (+15%)

### Optimistic Estimates (with tuning)
- **Precision**: 59.3% → 80% (+20.7%)
- **Recall**: 80.5% → 90% (+9.5%)
- **F1 Score**: 63.0% → 85% (+22%)

## Success Metrics

### Primary Metrics
1. **Precision@10**: % of recommended tools that are relevant
2. **Recall@10**: % of relevant tools that are recommended
3. **F1 Score**: Harmonic mean of precision and recall
4. **MRR (Mean Reciprocal Rank)**: Average 1/rank of first relevant tool

### Secondary Metrics
1. **Latency P99**: < 150ms for full pipeline
2. **Cache Hit Rate**: > 30% for cross-encoder
3. **Feature Coverage**: All features computed successfully
4. **Model Drift**: Monitor feature importance changes

## Risk Mitigation

### Performance Risks
- **Mitigation**: Async processing, caching, batch operations
- **Fallback**: Disable components if latency exceeds threshold

### Quality Risks
- **Mitigation**: A/B testing, gradual rollout
- **Fallback**: Revert to semantic-only if metrics degrade

### Operational Risks
- **Mitigation**: Feature flags for each component
- **Monitoring**: Real-time metrics dashboard

## Dependencies

### Python Packages
```toml
# Add to pyproject.toml
rank-bm25 = "^0.2.2"  # BM25 implementation
sentence-transformers = "^2.2.2"  # Cross-encoder
xgboost = "^2.0.0"  # Learning to rank
scikit-learn = "^1.3.0"  # Feature preprocessing
```

### Model Downloads
```bash
# Pre-download models
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

## Testing Strategy

### Unit Tests
- Test each component in isolation
- Mock external dependencies
- Validate feature extraction

### Integration Tests
- End-to-end pipeline testing
- Performance benchmarks
- Compatibility with existing code

### Evaluation Tests
- Run on ToolBench test set
- Compare metrics before/after
- Statistical significance testing

## Rollout Plan

### Week 1
- Implement and test BM25
- Deploy to staging
- Measure impact

### Week 2
- Add cross-encoder reranking
- Optimize for latency
- A/B test with 10% traffic

### Week 3
- Deploy LTR model
- Train on accumulated data
- Full rollout if metrics improve

## Monitoring & Observability

### Metrics to Track
```python
# Prometheus metrics
tool_filter_latency_histogram.observe(latency)
tool_filter_precision_gauge.set(precision)
tool_filter_recall_gauge.set(recall)
tool_filter_cache_hits_counter.inc()
```

### Logging
```python
logger.info("Pipeline metrics", extra={
    "bm25_score": bm25_score,
    "semantic_score": semantic_score,
    "ce_score": ce_score,
    "ltr_score": ltr_score,
    "final_score": final_score,
    "latency_ms": latency
})
```

### Alerts
- Latency > 200ms for P99
- Precision drops > 5%
- Cache hit rate < 20%
- Model prediction failures

## Future Enhancements

### Short Term (1-2 months)
1. Online learning for LTR model
2. Personalization based on user history
3. Multi-lingual support

### Medium Term (3-6 months)
1. Contextual bandits for exploration
2. Graph neural networks for tool relationships
3. Active learning for continuous improvement

### Long Term (6+ months)
1. Large language model fine-tuning
2. Multi-modal tool understanding (code + docs)
3. Reinforcement learning from user feedback

## References

1. [Robertson & Zaragoza, 2009] "The Probabilistic Relevance Framework: BM25 and Beyond"
2. [Nogueira & Cho, 2019] "Passage Re-ranking with BERT"
3. [Liu et al., 2009] "Learning to Rank for Information Retrieval"
4. [Burges, 2010] "From RankNet to LambdaRank to LambdaMART"
5. [Chen & Guestrin, 2016] "XGBoost: A Scalable Tree Boosting System"