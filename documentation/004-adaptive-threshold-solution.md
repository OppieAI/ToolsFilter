# Adaptive Threshold Solution for Hybrid Search

## Problem Analysis

### 1. The Score Gap Problem
- **Threshold Optimizer**: Reports 100% recall at threshold 0.438
- **Actual System**: Achieves only 50.6% recall
- **Root Cause**: Optimizer evaluates threshold-based selection, system uses rank-based selection

### 2. The Static Threshold Problem
- Using fixed threshold (0.438) for all queries ignores:
  - Query specificity variations
  - Available tools count variations
  - Score distribution differences
  - Semantic vs BM25 contribution ratios

## Solution: Adaptive Threshold Strategies

### Strategy 1: Statistical Adaptation
```python
# Adapt to current score distribution
threshold = mean(scores) - std(scores)  # Captures ~84% of normal distribution
```

**Pros**: Automatically adjusts to score distribution
**Cons**: Assumes normal distribution

### Strategy 2: Percentile-Based
```python
# Keep top X% of tools
threshold = percentile(scores, 80)  # Keep top 20%
```

**Pros**: Guarantees consistent proportion
**Cons**: May include poor tools if all scores are low

### Strategy 3: Gap Detection
```python
# Find natural clustering in scores
threshold = find_largest_gap(sorted_scores)
```

**Pros**: Finds natural quality boundaries
**Cons**: May not exist clear gap

### Strategy 4: Query-Aware Adjustment
```python
# Adjust based on query characteristics
threshold = base * query_length_factor * num_tools_factor
```

**Pros**: Context-aware
**Cons**: Requires tuning

## Recommended Implementation

### Hybrid Approach
1. **Primary**: Use rank-based selection (top-K)
2. **Secondary**: Apply minimum quality threshold
3. **Adaptive**: Adjust threshold per-query using statistical method

```python
def select_tools(scores, k=5, min_quality=0.3):
    # Step 1: Compute adaptive threshold
    adaptive_threshold = max(
        min_quality,  # Minimum quality floor
        np.percentile(scores, 70)  # Adaptive component
    )
    
    # Step 2: Filter by threshold
    qualified = [s for s in scores if s.score >= adaptive_threshold]
    
    # Step 3: Rank-based selection
    return sorted(qualified, key=lambda x: x.score, reverse=True)[:k]
```

## Configuration Changes

### Current (Static)
```python
primary_similarity_threshold: float = 0.438  # Fixed for all queries
```

### Proposed (Adaptive)
```python
threshold_strategy: str = "adaptive_statistical"  # or "percentile", "gap", "query_aware"
min_quality_threshold: float = 0.3  # Floor to ensure minimum quality
adaptive_percentile: float = 80  # For percentile strategy
```

## Expected Improvements

### Before (Static Threshold)
- Average Precision: 0.845
- Average Recall: 0.506
- F1 Score: 0.600

### After (Adaptive Threshold)
- Expected Precision: ~0.80 (slight decrease)
- Expected Recall: ~0.70 (significant increase)
- Expected F1: ~0.75 (overall improvement)

## Implementation Priority

1. **Phase 1**: Implement statistical adaptive threshold (quick win)
2. **Phase 2**: Add query-aware adjustments
3. **Phase 3**: Implement rank-based selection with diversity
4. **Phase 4**: A/B test different strategies

## Metrics to Track

1. **Score Distribution per Query**
   - Mean, std, min, max
   - Gap sizes
   - Clustering coefficient

2. **Threshold Adaptation**
   - How much does threshold vary?
   - Correlation with query length
   - Correlation with tool count

3. **Quality Metrics**
   - Precision/Recall/F1 per strategy
   - User satisfaction (if available)
   - Response time impact

## Code Integration Points

1. **VectorStoreService.hybrid_search()**
   - Replace fixed threshold with adaptive computation
   
2. **ThresholdOptimizer**
   - Add adaptive threshold evaluation
   - Compare static vs adaptive performance

3. **Config Settings**
   - Add strategy selection
   - Add strategy-specific parameters

## Conclusion

The static threshold approach is fundamentally mismatched with the dynamic nature of hybrid search scores. Adaptive thresholds that respond to:
- Current score distribution
- Query characteristics  
- Available tools count

Will provide more consistent and effective tool selection across diverse queries.