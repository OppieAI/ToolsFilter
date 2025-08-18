# Macro vs Micro Averaging in Information Retrieval

## The Bug We Fixed

The threshold optimizer was using **micro-averaging** (pooling all predictions) when it should use **macro-averaging** (per-query metrics, then average) to match industry standards and actual system behavior.

## Why Macro-Averaging is the Industry Standard

### 1. **Used by Major IR Benchmarks**
- **TREC** (Text REtrieval Conference): Uses macro-averaging
- **MS MARCO**: Macro-averages across queries
- **BEIR**: Benchmarks use macro-averaged nDCG@10
- **MTEB**: Massive Text Embedding Benchmark uses macro-averaging

### 2. **Reflects User Experience**
Each query is an independent user request. Users care about per-query performance, not aggregate statistics across all queries.

### 3. **Prevents Query Imbalance**
- Some queries might have 10 tools, others have 100
- Micro-averaging would let high-tool queries dominate
- Macro-averaging treats each query equally

## The Difference Illustrated

### Micro-Averaging (Wrong for IR)
```python
# Pool all predictions across all queries
all_predictions = [
    (query1_tool1, relevant=True, retrieved=True),
    (query1_tool2, relevant=False, retrieved=False),
    (query2_tool1, relevant=True, retrieved=False),  # Bad for Q2!
    (query2_tool2, relevant=False, retrieved=True),
]

# Global metrics
precision = 1/2 = 0.5
recall = 1/2 = 0.5
```

### Macro-Averaging (IR Standard)
```python
# Query 1 metrics
q1_precision = 1/1 = 1.0
q1_recall = 1/1 = 1.0

# Query 2 metrics  
q2_precision = 0/1 = 0.0
q2_recall = 0/1 = 0.0

# Average across queries
avg_precision = (1.0 + 0.0) / 2 = 0.5
avg_recall = (1.0 + 0.0) / 2 = 0.5
```

## Impact on Our System

### Before (Micro-averaging)
- Threshold optimizer: "100% recall at threshold 0.438"
- Actual system: 50.6% recall
- **Gap: 49.4%** (This was the bug!)

### After (Macro-averaging)
- Threshold optimizer: "~55% recall at threshold 0.438"
- Actual system: 50.6% recall
- **Gap: ~5%** (Much more accurate!)

## Implementation

```python
def calculate_metrics_at_threshold(self, threshold: float, use_macro_averaging: bool = True):
    """
    Args:
        use_macro_averaging: True for IR standard, False for legacy
    """
    if use_macro_averaging:
        # Group by query, calculate per-query metrics, then average
        # This matches how real IR systems are evaluated
    else:
        # Legacy micro-averaging (kept for backward compatibility)
```

## References

1. [TREC Evaluation](https://trec.nist.gov/pubs/trec15/appendices/CE.MEASURES06.pdf)
2. [MS MARCO Evaluation](https://microsoft.github.io/msmarco/)
3. [BEIR Benchmark](https://github.com/beir-cellar/beir)
4. [IR Evaluation Measures](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)

## Key Takeaway

**Always use macro-averaging for multi-query IR evaluation.** It's not just a preference—it's the industry standard that ensures fair evaluation across queries and matches real user experience.