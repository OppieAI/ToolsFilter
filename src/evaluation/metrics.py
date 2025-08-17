"""Evaluation metrics for PTR Tool Filter using RAGAS framework."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    precision_at_k: float
    recall_at_k: float
    f1_score: float
    mean_reciprocal_rank: float
    ndcg_score: float
    average_similarity: float
    processing_time_ms: float
    timestamp: str
    metadata: Dict[str, Any]


class ToolFilterEvaluator:
    """Evaluator for tool filtering performance using RAGAS-inspired metrics."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics_history = []
    
    def evaluate_tool_recommendations(
        self,
        predicted_tools: List[str],
        ground_truth_tools: List[str],
        similarity_scores: List[float],
        processing_time_ms: float,
        k: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationResult:
        """
        Evaluate tool recommendations against ground truth.
        
        Args:
            predicted_tools: List of recommended tool names
            ground_truth_tools: List of expected tool names
            similarity_scores: Similarity scores for each predicted tool
            processing_time_ms: Processing time in milliseconds
            k: Number of top recommendations to consider
            metadata: Additional metadata
            
        Returns:
            EvaluationResult with computed metrics
        """
        # Limit to top-k predictions
        predicted_tools = predicted_tools[:k]
        similarity_scores = similarity_scores[:k]
        
        # Calculate precision@k
        precision_at_k = self._calculate_precision_at_k(
            predicted_tools, ground_truth_tools
        )
        
        # Calculate recall@k
        recall_at_k = self._calculate_recall_at_k(
            predicted_tools, ground_truth_tools
        )
        
        # Calculate F1 score
        f1_score = self._calculate_f1_score(precision_at_k, recall_at_k)
        
        # Calculate Mean Reciprocal Rank (MRR)
        mrr = self._calculate_mrr(predicted_tools, ground_truth_tools)
        
        # Calculate NDCG
        ndcg = self._calculate_ndcg(
            predicted_tools, ground_truth_tools, similarity_scores
        )
        
        # Calculate average similarity score
        avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        result = EvaluationResult(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_score=f1_score,
            mean_reciprocal_rank=mrr,
            ndcg_score=ndcg,
            average_similarity=avg_similarity,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )
        
        # Store in history
        self.metrics_history.append(result)
        
        return result
    
    def _calculate_precision_at_k(
        self, 
        predicted: List[str], 
        ground_truth: List[str]
    ) -> float:
        """Calculate precision@k metric."""
        if not predicted:
            return 0.0
        
        relevant_predicted = [p for p in predicted if p in ground_truth]
        return len(relevant_predicted) / len(predicted)
    
    def _calculate_recall_at_k(
        self,
        predicted: List[str],
        ground_truth: List[str]
    ) -> float:
        """Calculate recall@k metric."""
        if not ground_truth:
            return 1.0 if not predicted else 0.0
        
        relevant_predicted = [p for p in predicted if p in ground_truth]
        return len(relevant_predicted) / len(ground_truth)
    
    def _calculate_f1_score(
        self,
        precision: float,
        recall: float
    ) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def _calculate_mrr(
        self,
        predicted: List[str],
        ground_truth: List[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, tool in enumerate(predicted):
            if tool in ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(
        self,
        predicted: List[str],
        ground_truth: List[str],
        scores: List[float]
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain."""
        if not predicted or not ground_truth:
            return 0.0
        
        # Create relevance scores (1 if in ground truth, 0 otherwise)
        relevance = [1.0 if tool in ground_truth else 0.0 for tool in predicted]
        
        # Calculate DCG
        dcg = sum(
            rel / np.log2(i + 2) 
            for i, rel in enumerate(relevance)
        )
        
        # Calculate ideal DCG
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(
            rel / np.log2(i + 2) 
            for i, rel in enumerate(ideal_relevance)
        )
        
        # Calculate NDCG
        return dcg / idcg if idcg > 0 else 0.0
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from evaluation history."""
        if not self.metrics_history:
            return {}
        
        metrics = {
            "avg_precision": np.mean([r.precision_at_k for r in self.metrics_history]),
            "avg_recall": np.mean([r.recall_at_k for r in self.metrics_history]),
            "avg_f1": np.mean([r.f1_score for r in self.metrics_history]),
            "avg_mrr": np.mean([r.mean_reciprocal_rank for r in self.metrics_history]),
            "avg_ndcg": np.mean([r.ndcg_score for r in self.metrics_history]),
            "avg_similarity": np.mean([r.average_similarity for r in self.metrics_history]),
            "avg_processing_time_ms": np.mean([r.processing_time_ms for r in self.metrics_history]),
            "p95_processing_time_ms": np.percentile([r.processing_time_ms for r in self.metrics_history], 95),
            "total_evaluations": len(self.metrics_history)
        }
        
        return metrics
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        metrics_data = {
            "evaluations": [
                {
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "f1_score": r.f1_score,
                    "mean_reciprocal_rank": r.mean_reciprocal_rank,
                    "ndcg_score": r.ndcg_score,
                    "average_similarity": r.average_similarity,
                    "processing_time_ms": r.processing_time_ms,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                }
                for r in self.metrics_history
            ],
            "aggregate_metrics": self.get_aggregate_metrics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Exported {len(self.metrics_history)} evaluations to {filepath}")


# Test data for evaluation
EVALUATION_TEST_CASES = [
    {
        "name": "Find Python files",
        "messages": [
            {"role": "user", "content": "I need to find all Python files in the project"}
        ],
        "expected_tools": ["find", "grep", "ls"],
        "description": "User wants to search for Python files"
    },
    {
        "name": "Debug error",
        "messages": [
            {"role": "user", "content": "I'm getting an error in my code"},
            {"role": "assistant", "content": "I'll help you debug the error. Can you share the error message?"},
            {"role": "user", "content": "TypeError: cannot read property 'map' of undefined"}
        ],
        "expected_tools": ["grep", "cat", "tail"],
        "description": "User needs help debugging an error"
    },
    {
        "name": "Create directory",
        "messages": [
            {"role": "user", "content": "Create a new directory called 'src/components'"}
        ],
        "expected_tools": ["mkdir", "ls", "pwd"],
        "description": "User wants to create a directory"
    },
    {
        "name": "File operations",
        "messages": [
            {"role": "user", "content": "Copy all .js files from src to backup folder"}
        ],
        "expected_tools": ["cp", "find", "ls", "mkdir"],
        "description": "User wants to copy files"
    },
    {
        "name": "Search pattern",
        "messages": [
            {"role": "user", "content": "Search for all occurrences of 'TODO' in the codebase"}
        ],
        "expected_tools": ["grep", "find", "sed"],
        "description": "User wants to search for a pattern"
    }
]