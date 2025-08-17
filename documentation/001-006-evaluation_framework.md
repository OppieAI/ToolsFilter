# PTR Evaluation Framework

## Overview
A comprehensive evaluation setup for the PTR tool filtering system using best-in-class open source libraries.

## Core Libraries

### 1. RAGAS - Primary Evaluation Metrics
```bash
pip install ragas
```

### 2. Phoenix - Observability & Debugging
```bash
pip install arize-phoenix
```

### 3. MLflow - Experiment Tracking
```bash
pip install mlflow
```

## Evaluation Metrics for PTR

### 1. Precision Metrics
- **Precision@K**: Of the top K recommended tools, how many were actually needed?
- **Tool Precision**: Percentage of recommended tools that were used

### 2. Recall Metrics
- **Recall@K**: Of all needed tools, how many were in the top K recommendations?
- **Coverage**: Did we recommend all necessary tools?

### 3. Ranking Metrics
- **NDCG** (Normalized Discounted Cumulative Gain): Quality of ranking
- **MRR** (Mean Reciprocal Rank): Position of first relevant tool

### 4. Performance Metrics
- **Latency**: Time to generate recommendations
- **Throughput**: Requests per second

## Implementation Example

```python
from dataclasses import dataclass
from typing import List, Dict, Set
import time
import mlflow
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
import phoenix as px
import numpy as np

@dataclass
class EvalCase:
    query: str
    messages: List[Dict[str, str]]
    expected_tools: Set[str]
    all_available_tools: List[str]

class PTREvaluator:
    def __init__(self, ptr_system):
        self.ptr_system = ptr_system
        self.phoenix_session = px.launch_app()
        
    def evaluate_single(self, eval_case: EvalCase) -> Dict:
        """Evaluate a single test case"""
        start_time = time.time()
        
        # Get recommendations
        recommended_tools = self.ptr_system.filter_tools(
            messages=eval_case.messages,
            available_tools=eval_case.all_available_tools
        )
        
        latency = (time.time() - start_time) * 1000
        
        # Calculate metrics
        recommended_set = set([t['tool_name'] for t in recommended_tools[:10]])
        expected_set = eval_case.expected_tools
        
        # Precision: What % of recommended tools were correct?
        precision = len(recommended_set & expected_set) / len(recommended_set) if recommended_set else 0
        
        # Recall: What % of expected tools were recommended?
        recall = len(recommended_set & expected_set) / len(expected_set) if expected_set else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Ranking metrics
        ranking_scores = self._calculate_ranking_metrics(recommended_tools, expected_set)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'latency_ms': latency,
            'ndcg': ranking_scores['ndcg'],
            'mrr': ranking_scores['mrr'],
            'recommended': list(recommended_set),
            'expected': list(expected_set)
        }
    
    def _calculate_ranking_metrics(self, recommended_tools: List[Dict], expected_set: Set[str]) -> Dict:
        """Calculate NDCG and MRR"""
        # MRR - position of first relevant result
        mrr = 0
        for i, tool in enumerate(recommended_tools):
            if tool['tool_name'] in expected_set:
                mrr = 1 / (i + 1)
                break
        
        # NDCG - quality of ranking
        relevance_scores = [1 if t['tool_name'] in expected_set else 0 for t in recommended_tools[:10]]
        if sum(relevance_scores) == 0:
            ndcg = 0
        else:
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
            ideal_relevance = sorted(relevance_scores, reverse=True)
            idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
            ndcg = dcg / idcg if idcg > 0 else 0
        
        return {'ndcg': ndcg, 'mrr': mrr}
    
    def run_evaluation_suite(self, test_cases: List[EvalCase], experiment_name: str):
        """Run full evaluation suite with MLflow tracking"""
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log configuration
            mlflow.log_param("embedding_model", self.ptr_system.embedding_model)
            mlflow.log_param("num_test_cases", len(test_cases))
            
            # Run evaluations
            results = []
            for case in test_cases:
                with px.using_project("ptr-eval"):
                    result = self.evaluate_single(case)
                    results.append(result)
            
            # Aggregate metrics
            avg_metrics = {
                'avg_precision': np.mean([r['precision'] for r in results]),
                'avg_recall': np.mean([r['recall'] for r in results]),
                'avg_f1': np.mean([r['f1'] for r in results]),
                'avg_latency_ms': np.mean([r['latency_ms'] for r in results]),
                'p95_latency_ms': np.percentile([r['latency_ms'] for r in results], 95),
                'avg_ndcg': np.mean([r['ndcg'] for r in results]),
                'avg_mrr': np.mean([r['mrr'] for r in results])
            }
            
            # Log to MLflow
            for metric_name, value in avg_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Use RAGAS for additional evaluation
            self._ragas_evaluation(test_cases, results)
            
            print(f"Evaluation Results for {experiment_name}:")
            for k, v in avg_metrics.items():
                print(f"{k}: {v:.3f}")
            
            return avg_metrics
    
    def _ragas_evaluation(self, test_cases: List[EvalCase], results: List[Dict]):
        """Additional evaluation using RAGAS"""
        # Format data for RAGAS
        data = {
            'question': [case.query for case in test_cases],
            'contexts': [[t['tool_name'] for t in r['recommended']] for r in results],
            'ground_truth': [list(case.expected_tools) for case in test_cases]
        }
        
        # Evaluate with RAGAS
        ragas_results = evaluate(
            dataset=data,
            metrics=[context_precision, context_recall]
        )
        
        # Log RAGAS metrics
        for metric, value in ragas_results.items():
            mlflow.log_metric(f"ragas_{metric}", value)

# Example usage
def create_test_cases() -> List[EvalCase]:
    """Create test cases for evaluation"""
    return [
        EvalCase(
            query="Find all Python files",
            messages=[
                {"role": "user", "content": "I need to find all Python files in the project"},
                {"role": "assistant", "content": "I'll help you find Python files"}
            ],
            expected_tools={"glob", "find", "grep"},
            all_available_tools=["glob", "find", "grep", "ls", "cat", "sed", "awk", "rm", "cp"]
        ),
        EvalCase(
            query="Debug API errors",
            messages=[
                {"role": "user", "content": "The API is returning 500 errors"},
                {"role": "assistant", "content": "Let me help debug the API errors"}
            ],
            expected_tools={"curl", "logs", "grep", "tail"},
            all_available_tools=["curl", "logs", "grep", "tail", "ping", "ssh", "docker", "kubectl"]
        )
    ]

# Run evaluation
if __name__ == "__main__":
    # Initialize your PTR system
    ptr_system = PTRSystem(embedding_model="voyage-2")
    
    # Create evaluator
    evaluator = PTREvaluator(ptr_system)
    
    # Create test cases
    test_cases = create_test_cases()
    
    # Run evaluation
    results = evaluator.run_evaluation_suite(
        test_cases=test_cases,
        experiment_name="PTR_Voyage_Embedding_Eval"
    )
```

## A/B Testing Different Approaches

```python
def ab_test_embeddings():
    """Compare different embedding models"""
    models_to_test = ["text-embedding-3-small", "voyage-2", "embed-english-v3.0"]
    
    test_cases = create_test_cases()
    
    for model in models_to_test:
        # Configure PTR with different embedding
        ptr_system = PTRSystem(embedding_model=model)
        evaluator = PTREvaluator(ptr_system)
        
        # Run evaluation
        results = evaluator.run_evaluation_suite(
            test_cases=test_cases,
            experiment_name=f"PTR_AB_Test_{model}"
        )
    
    # Compare results in MLflow UI
    print("View results at: http://localhost:5000")
    mlflow.ui.run()
```

## Continuous Evaluation Pipeline

```python
import schedule
import time

def continuous_evaluation():
    """Run evaluations periodically"""
    def run_eval():
        ptr_system = PTRSystem()
        evaluator = PTREvaluator(ptr_system)
        
        # Get latest test cases (could be from a database)
        test_cases = get_latest_test_cases()
        
        # Run evaluation
        results = evaluator.run_evaluation_suite(
            test_cases=test_cases,
            experiment_name=f"PTR_Daily_Eval_{time.strftime('%Y%m%d')}"
        )
        
        # Alert if performance degrades
        if results['avg_f1'] < 0.7:
            send_alert("PTR performance degraded below threshold")
    
    # Schedule daily evaluation
    schedule.every().day.at("02:00").do(run_eval)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
```

## Best Practices

1. **Create Diverse Test Cases**
   - Cover different query types
   - Include edge cases
   - Test with varying numbers of available tools

2. **Track Multiple Metrics**
   - Don't optimize for just one metric
   - Balance precision and recall
   - Monitor latency alongside quality

3. **Version Your Test Sets**
   - Keep test cases in version control
   - Document why each case exists
   - Update as you discover new patterns

4. **Automate Evaluation**
   - Run on every deployment
   - Set up alerts for regression
   - Track trends over time

5. **Use Phoenix for Debugging**
   - When metrics drop, use Phoenix to understand why
   - Trace individual recommendations
   - Identify patterns in failures