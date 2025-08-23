# Tool Filtering Performance Improvement Roadmap

## Current Performance Analysis

### Baseline (Without Noise)
- **MRR**: 0.875 (Excellent)
- **P@1**: 0.750 (Good)
- **Expected Recall**: 100% (Perfect)
- **Average first relevant rank**: 1.25

### With Realistic Noise (100 tools)
- **MRR**: 0.395 (Poor - 55% degradation)
- **P@1**: 0.100 (Critical - 87% degradation)
- **Expected Recall**: 92.5% (Good)
- **Average first relevant rank**: 3.10
- **Noise in results**: 61% (Too high)

## Performance Goals

To achieve production-ready quality, we need:
- **MRR > 0.6** (first relevant result in top 2 positions)
- **P@1 > 0.5** (50% accuracy at position 1)
- **Noise proportion < 30%** (cleaner results)
- **Expected recall > 90%** (maintain current level)

## Improvement Strategies

### 1. Fix LTR Feature Extraction (HIGHEST PRIORITY)

#### Problem
Current features are constants that don't vary with queries:
```python
# Current problematic features
'domain_network': 71.24% importance  # Always same for similar tools
'domain_api': high importance        # Constant across tool types
```

#### Solution
Replace with query-tool interaction features:

```python
def extract_improved_features(query: str, tool: Tool) -> Dict[str, float]:
    """Extract meaningful features that capture query-tool relationships."""
    
    query_tokens = set(query.lower().split())
    tool_tokens = set(tool.name.lower().split('_'))
    desc_tokens = set(tool.description.lower().split())
    
    features = {
        # Lexical overlap features
        'query_tool_name_overlap': len(query_tokens & tool_tokens) / max(len(query_tokens), 1),
        'query_desc_overlap': len(query_tokens & desc_tokens) / max(len(query_tokens), 1),
        'jaccard_similarity': len(query_tokens & desc_tokens) / len(query_tokens | desc_tokens),
        
        # Semantic features (requires embeddings)
        'query_tool_cosine_sim': cosine_similarity(query_embedding, tool_embedding),
        'query_desc_cosine_sim': cosine_similarity(query_embedding, desc_embedding),
        
        # Intent matching
        'verb_match': float(any(verb in tool.name.lower() for verb in extract_verbs(query))),
        'action_alignment': calculate_action_alignment(query, tool),
        
        # Length ratios
        'query_length_ratio': len(query) / max(len(tool.description), 1),
        'name_length_ratio': len(query) / max(len(tool.name), 1),
        
        # Parameter matching
        'param_relevance': calculate_param_relevance(query, tool.parameters),
        'required_param_match': match_required_params(query, tool.parameters.get('required', [])),
        
        # Position features (from initial ranking)
        'initial_rank': initial_rank,
        'initial_score': initial_score,
        'score_diff_from_top': top_score - initial_score,
    }
    
    return features
```

### 2. Implement Two-Stage Filtering

```python
class TwoStageSearchService:
    """Implement aggressive filtering followed by precise reranking."""
    
    async def search(self, query: str, available_tools: List[Tool]) -> List[Dict]:
        # Stage 1: Cast wide net with lower threshold
        initial_candidates = await self.broad_search(
            query=query,
            tools=available_tools,
            threshold=0.10,  # Lower threshold
            limit=50  # Get more candidates
        )
        
        # Stage 2: Precise reranking with strict criteria
        final_results = await self.precise_rerank(
            query=query,
            candidates=initial_candidates,
            threshold=0.15,  # Stricter threshold
            limit=10  # Return only best
        )
        
        return final_results
    
    async def precise_rerank(self, query: str, candidates: List[Dict], 
                            threshold: float, limit: int) -> List[Dict]:
        """Apply multiple reranking strategies."""
        
        # Cross-encoder reranking
        if self.enable_cross_encoder:
            candidates = await self.cross_encoder_rerank(query, candidates)
        
        # LTR reranking with fixed features
        if self.enable_ltr:
            candidates = await self.ltr_rerank(query, candidates)
        
        # Filter by confidence
        candidates = [c for c in candidates if c['score'] >= threshold]
        
        # Apply confidence cutoff
        candidates = self.apply_confidence_cutoff(candidates)
        
        return candidates[:limit]
```

### 3. Adaptive Threshold Strategy

```python
class AdaptiveThresholdManager:
    """Dynamically adjust thresholds based on context."""
    
    def get_threshold(self, 
                     query: str, 
                     num_available_tools: int,
                     query_complexity: str = None) -> float:
        """Calculate context-aware threshold."""
        
        base_threshold = 0.13
        adjustments = []
        
        # Query length adjustment
        query_words = len(query.split())
        if query_words < 5:
            adjustments.append(0.03)  # Simple query, be stricter
        elif query_words > 20:
            adjustments.append(-0.02)  # Complex query, be looser
        
        # Tool count adjustment
        if num_available_tools > 100:
            adjustments.append(0.02)  # Many tools, be stricter
        elif num_available_tools < 20:
            adjustments.append(-0.01)  # Few tools, be looser
        
        # Query type adjustment
        if query_complexity == 'simple':  # e.g., "get user"
            adjustments.append(0.02)
        elif query_complexity == 'complex':  # e.g., multi-step operations
            adjustments.append(-0.02)
        
        # Domain-specific adjustment
        if self.is_technical_domain(query):
            adjustments.append(0.01)  # Technical queries need precision
        
        final_threshold = base_threshold + sum(adjustments)
        return max(0.05, min(0.25, final_threshold))  # Clamp to reasonable range
```

### 4. Query-Aware Filtering

```python
class QueryAwareFilter:
    """Filter tools based on query intent and context."""
    
    def filter_by_intent(self, query: str, tools: List[Tool]) -> List[Tool]:
        """Filter tools that match query intent."""
        
        # Extract query intent
        intent = self.extract_intent(query)
        action_verbs = self.extract_action_verbs(query)
        entities = self.extract_entities(query)
        
        filtered_tools = []
        for tool in tools:
            # Check action alignment
            if not self.matches_action(tool, action_verbs):
                continue
            
            # Check entity relevance
            if entities and not self.has_relevant_params(tool, entities):
                continue
            
            # Check domain compatibility
            if not self.compatible_domain(tool, intent):
                continue
            
            filtered_tools.append(tool)
        
        return filtered_tools
    
    def extract_intent(self, query: str) -> Dict[str, Any]:
        """Extract intent from query."""
        patterns = {
            'create': r'\b(create|add|new|generate|make)\b',
            'read': r'\b(get|fetch|retrieve|find|search|list|show)\b',
            'update': r'\b(update|modify|change|edit|set)\b',
            'delete': r'\b(delete|remove|clear|drop)\b',
        }
        
        intent = {'crud_op': None, 'domain': None}
        for op, pattern in patterns.items():
            if re.search(pattern, query.lower()):
                intent['crud_op'] = op
                break
        
        return intent
```

### 5. Ensemble Scoring System

```python
class EnsembleScorer:
    """Combine multiple scoring signals with learned weights."""
    
    def __init__(self):
        # Weights learned from validation data
        self.weights = {
            'semantic': 0.40,
            'bm25': 0.25,
            'exact_match': 0.15,
            'cross_encoder': 0.20
        }
    
    async def score(self, query: str, tool: Tool) -> float:
        """Calculate ensemble score for query-tool pair."""
        
        scores = {}
        
        # Semantic similarity
        scores['semantic'] = await self.semantic_similarity(query, tool)
        
        # BM25 score
        scores['bm25'] = self.bm25_score(query, tool)
        
        # Exact match bonuses
        scores['exact_match'] = self.exact_match_score(query, tool)
        
        # Cross-encoder score (if available)
        if self.cross_encoder:
            scores['cross_encoder'] = await self.cross_encoder_score(query, tool)
        else:
            scores['cross_encoder'] = scores['semantic']  # Fallback
        
        # Weighted combination
        final_score = sum(scores[k] * self.weights[k] for k in scores)
        
        # Apply confidence penalties
        if self.is_low_confidence(scores):
            final_score *= 0.7
        
        return final_score
    
    def is_low_confidence(self, scores: Dict[str, float]) -> bool:
        """Check if scores indicate low confidence."""
        # High disagreement between methods
        variance = np.var(list(scores.values()))
        if variance > 0.3:
            return True
        
        # All scores are mediocre
        if all(s < 0.5 for s in scores.values()):
            return True
        
        return False
```

### 6. Negative Filtering

```python
class NegativeFilter:
    """Remove obvious non-matches early in the pipeline."""
    
    def filter(self, query: str, tools: List[Tool]) -> List[Tool]:
        """Apply negative filtering rules."""
        
        # Extract negative context
        negative_terms = self.extract_negatives(query)
        must_not_have = self.extract_exclusions(query)
        
        filtered = []
        for tool in tools:
            # Skip if tool matches negative patterns
            if negative_terms and self.matches_negative(tool, negative_terms):
                continue
            
            # Skip if tool has excluded features
            if must_not_have and self.has_excluded_features(tool, must_not_have):
                continue
            
            # Skip if domain mismatch
            if not self.domain_compatible(query, tool):
                continue
            
            filtered.append(tool)
        
        return filtered
    
    def extract_negatives(self, query: str) -> List[str]:
        """Extract negative indicators from query."""
        patterns = [
            r'not?\s+(\w+)',
            r'except\s+(\w+)',
            r'without\s+(\w+)',
            r'exclude\s+(\w+)'
        ]
        
        negatives = []
        for pattern in patterns:
            matches = re.findall(pattern, query.lower())
            negatives.extend(matches)
        
        return negatives
```

### 7. Confidence-Based Cutoff

```python
class ConfidenceCutoff:
    """Stop returning results when confidence drops."""
    
    def apply_cutoff(self, results: List[Dict]) -> List[Dict]:
        """Apply confidence-based cutoff to results."""
        
        if not results:
            return []
        
        filtered = [results[0]]  # Always include top result
        top_score = results[0]['score']
        
        for i in range(1, len(results)):
            # Stop if score drops too much
            if results[i]['score'] < 0.6 * top_score:
                break
            
            # Stop if below minimum confidence
            if results[i]['score'] < 0.15:
                break
            
            # Stop if large gap from previous
            if results[i-1]['score'] - results[i]['score'] > 0.1:
                if i > 3:  # Allow at least 3 results
                    break
            
            filtered.append(results[i])
        
        return filtered[:10]  # Hard limit
```

### 8. Configuration Optimization

```python
# config.py adjustments
class Settings(BaseSettings):
    # Threshold adjustments
    primary_similarity_threshold: float = 0.15  # Increase from 0.13
    cross_encoder_threshold: float = 0.5  # Strict reranking threshold
    
    # Pipeline configuration
    enable_two_stage_search: bool = True
    max_initial_candidates: int = 50  # Cast wider net
    final_top_k: int = 10  # Return fewer, high-quality results
    
    # Feature flags
    enable_adaptive_threshold: bool = True
    enable_query_aware_filtering: bool = True
    enable_ensemble_scoring: bool = True
    enable_negative_filtering: bool = True
    enable_confidence_cutoff: bool = True
    
    # LTR configuration
    ltr_use_improved_features: bool = True
    ltr_min_training_samples: int = 100
```

## Implementation Timeline

### Week 1 (Critical Fixes)
1. **Fix LTR Features** (2 days)
   - Remove constant domain features
   - Add query-tool interaction features
   - Test with validation set

2. **Two-Stage Filtering** (2 days)
   - Implement broad search + precise rerank
   - Tune stage thresholds
   - Measure noise reduction

3. **Configuration Tuning** (1 day)
   - Adjust thresholds based on test results
   - Enable/disable features based on performance

### Week 2 (Enhancements)
1. **Adaptive Thresholds** (2 days)
   - Implement context-aware threshold adjustment
   - Test on various query types

2. **Ensemble Scoring** (2 days)
   - Combine multiple scoring signals
   - Learn optimal weights from validation data

3. **Query-Aware Filtering** (1 day)
   - Implement intent extraction
   - Add domain compatibility checks

### Week 3 (Optimization)
1. **Negative Filtering** (1 day)
   - Implement exclusion rules
   - Test edge cases

2. **Confidence Cutoff** (1 day)
   - Implement dynamic result limiting
   - Tune cutoff parameters

3. **Performance Testing** (3 days)
   - Full evaluation with noise
   - A/B testing if possible
   - Fine-tune all parameters

## Success Metrics

### Primary Goals
- [ ] MRR > 0.6 with 100 noise tools
- [ ] P@1 > 0.5 with 100 noise tools
- [ ] Noise proportion < 30% in results
- [ ] Maintain recall > 90%

### Secondary Goals
- [ ] Average response time < 200ms
- [ ] NDCG@10 > 0.7
- [ ] User satisfaction score > 4/5 (if measurable)

## Testing Strategy

1. **Unit Tests**: Each component independently
2. **Integration Tests**: Full pipeline with sample data
3. **Evaluation Suite**: ToolBench evaluator with noise
4. **A/B Testing**: If possible, compare old vs new in production
5. **User Studies**: Collect feedback on result quality

## Monitoring

After deployment, monitor:
- Query latency (p50, p95, p99)
- Result quality metrics (MRR, P@k, NDCG)
- User engagement (click-through rate on top results)
- Error rates and edge cases

## Rollback Plan

If performance degrades:
1. Quick revert to previous configuration
2. Disable new features via feature flags
3. Investigate issues with increased logging
4. Fix and re-deploy with careful monitoring