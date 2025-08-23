"""
Metrics calculation engine for the evaluation framework.
Centralizes all metric computations with support for various metric types.
"""

import math
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict

from .models import EvaluationResult, TestCase, EvaluationRun, MetricValue
from .config import MetricsConfig


class MetricType(Enum):
    """Types of metrics available for calculation."""
    # Traditional metrics
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    ACCURACY = "accuracy"
    
    # Ranking metrics
    MRR = "mrr"  # Mean Reciprocal Rank
    NDCG = "ndcg"  # Normalized Discounted Cumulative Gain
    MAP = "map"  # Mean Average Precision
    PRECISION_AT_K = "precision_at_k"
    RECALL_AT_K = "recall_at_k"
    
    # Noise impact metrics
    NOISE_PROPORTION = "noise_proportion"
    EXPECTED_TOOL_RECALL = "expected_tool_recall"
    NOISE_RESISTANCE = "noise_resistance"
    
    # Performance metrics
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    
    # Statistical metrics
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"
    PERCENTILE = "percentile"


@dataclass
class MetricResult:
    """Result of a metric calculation."""
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metric": self.metric_type.value,
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple test results."""
    metrics: Dict[str, float]
    per_test_metrics: List[Dict[str, float]]
    statistical_summary: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "aggregated": self.metrics,
            "per_test": self.per_test_metrics,
            "statistics": self.statistical_summary,
            "metadata": self.metadata
        }
    
    def get_metric(self, metric_name: str) -> Optional[float]:
        """Get a specific metric value."""
        return self.metrics.get(metric_name)


class MetricsCalculator:
    """
    Main metrics calculation engine.
    Computes various types of metrics for evaluation results.
    """
    
    def __init__(self, config: MetricsConfig):
        """
        Initialize the metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config
        
    def calculate_metrics_for_result(
        self,
        result: EvaluationResult,
        noise_tool_names: Optional[Set[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate all metrics for a single evaluation result.
        
        Args:
            result: Evaluation result
            noise_tool_names: Optional set of noise tool names
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        # Get tool sets
        expected_set = set(result.test_case.expected_tools)
        recommended_tools = result.recommended_tools
        recommended_names = []
        for tool in recommended_tools:
            # Try both 'tool_name' and 'name' fields
            tool_name = tool.get("tool_name") or tool.get("name")
            if tool_name:
                recommended_names.append(tool_name)
        recommended_set = set(recommended_names)
        
        # Traditional metrics
        if self.config.calculate_traditional_metrics:
            traditional = self._calculate_traditional_metrics(
                expected_set, recommended_set
            )
            metrics.update(traditional)
        
        # Ranking metrics
        if self.config.calculate_ranking_metrics:
            ranking = self._calculate_ranking_metrics(
                expected_set, recommended_names
            )
            metrics.update(ranking)
        
        # Noise impact metrics
        if self.config.calculate_noise_metrics and noise_tool_names:
            noise = self._calculate_noise_metrics(
                expected_set, recommended_names, noise_tool_names
            )
            metrics.update(noise)
        
        # Performance metrics
        metrics["execution_time_ms"] = result.execution_time_ms
        
        return metrics
    
    def _calculate_traditional_metrics(
        self,
        expected_set: Set[str],
        recommended_set: Set[str]
    ) -> Dict[str, float]:
        """
        Calculate traditional IR metrics.
        
        Args:
            expected_set: Set of expected tool names
            recommended_set: Set of recommended tool names
            
        Returns:
            Dictionary of traditional metrics
        """
        if not expected_set:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0
            }
        
        true_positives = len(expected_set & recommended_set)
        false_positives = len(recommended_set - expected_set)
        false_negatives = len(expected_set - recommended_set)
        
        # Precision: Of items returned, how many are relevant?
        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        
        # Recall: Of relevant items, how many were returned?
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        
        # F1 Score: Harmonic mean of precision and recall
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        # Accuracy (less meaningful for IR but included for completeness)
        # Note: We don't have true negatives in IR context
        accuracy = precision  # Simplified accuracy
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def _calculate_ranking_metrics(
        self,
        expected_set: Set[str],
        recommended_list: List[str]
    ) -> Dict[str, float]:
        """
        Calculate ranking-aware metrics.
        
        Args:
            expected_set: Set of expected tool names
            recommended_list: Ordered list of recommended tool names
            
        Returns:
            Dictionary of ranking metrics
        """
        if not expected_set or not recommended_list:
            return self._empty_ranking_metrics()
        
        metrics = {}
        
        # Mean Reciprocal Rank (MRR)
        mrr = 0.0
        first_relevant_rank = None
        for i, tool_name in enumerate(recommended_list):
            if tool_name in expected_set:
                first_relevant_rank = i + 1  # 1-indexed
                mrr = 1.0 / first_relevant_rank
                break
        
        metrics["mrr"] = mrr
        metrics["first_relevant_rank"] = first_relevant_rank if first_relevant_rank else float('inf')
        
        # NDCG at different cutoffs
        for k in self.config.ndcg_cutoffs:
            ndcg_k = self._calculate_ndcg_at_k(expected_set, recommended_list, k)
            metrics[f"ndcg@{k}"] = ndcg_k
        
        # Precision at k
        for k in self.config.precision_cutoffs:
            p_at_k = self._calculate_precision_at_k(expected_set, recommended_list, k)
            metrics[f"p@{k}"] = p_at_k
            
            # Also calculate recall at k
            r_at_k = self._calculate_recall_at_k(expected_set, recommended_list, k)
            metrics[f"r@{k}"] = r_at_k
        
        # Mean Average Precision (MAP)
        map_score = self._calculate_map(expected_set, recommended_list)
        metrics["map"] = map_score
        
        # Count of expected tools in top positions
        for k in [3, 5, 10]:
            count = sum(1 for t in recommended_list[:k] if t in expected_set)
            metrics[f"expected_in_top{k}"] = count
        
        return metrics
    
    def _calculate_ndcg_at_k(
        self,
        expected_set: Set[str],
        recommended_list: List[str],
        k: int
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            expected_set: Set of expected tools
            recommended_list: Ordered list of recommendations
            k: Cutoff position
            
        Returns:
            NDCG@k score
        """
        import logging
        logger = logging.getLogger(__name__)
        
        dcg = 0.0
        idcg = 0.0
        
        # Calculate DCG@k
        relevant_positions = []
        for i in range(min(k, len(recommended_list))):
            if recommended_list[i] in expected_set:
                relevance = 1.0  # Binary relevance
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1)=0
                relevant_positions.append(i + 1)  # 1-indexed for logging
        
        # Calculate ideal DCG@k
        num_expected = len(expected_set)
        for i in range(min(k, num_expected)):
            idcg += 1.0 / math.log2(i + 2)
        
        # NDCG = DCG / IDCG
        ndcg_result = dcg / idcg if idcg > 0 else 0.0
        
        # Debug logging for NDCG@5 specifically
        if k == 5:
            logger.debug(f"NDCG@{k} calculation:")
            logger.debug(f"  Expected tools: {list(expected_set)}")
            logger.debug(f"  Recommended (first {k}): {recommended_list[:k]}")
            logger.debug(f"  Relevant positions: {relevant_positions}")
            logger.debug(f"  DCG: {dcg:.6f}, IDCG: {idcg:.6f}, NDCG: {ndcg_result:.6f}")
            
            # Check for potential issues
            if len(recommended_list) == 0:
                logger.debug("  WARNING: Empty recommended list!")
            if len(expected_set) == 0:
                logger.debug("  WARNING: Empty expected set!")
            if dcg > 0 and ndcg_result == 0:
                logger.debug("  ERROR: DCG > 0 but NDCG = 0!")
        
        return ndcg_result
    
    def _calculate_precision_at_k(
        self,
        expected_set: Set[str],
        recommended_list: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision at k.
        
        Args:
            expected_set: Set of expected tools
            recommended_list: Ordered list of recommendations
            k: Cutoff position
            
        Returns:
            P@k score
        """
        top_k = recommended_list[:k]
        if not top_k:
            return 0.0
        
        relevant_in_top_k = sum(1 for t in top_k if t in expected_set)
        return relevant_in_top_k / len(top_k)
    
    def _calculate_recall_at_k(
        self,
        expected_set: Set[str],
        recommended_list: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall at k.
        
        Args:
            expected_set: Set of expected tools
            recommended_list: Ordered list of recommendations
            k: Cutoff position
            
        Returns:
            R@k score
        """
        if not expected_set:
            return 0.0
        
        top_k = recommended_list[:k]
        relevant_in_top_k = sum(1 for t in top_k if t in expected_set)
        return relevant_in_top_k / len(expected_set)
    
    def _calculate_map(
        self,
        expected_set: Set[str],
        recommended_list: List[str]
    ) -> float:
        """
        Calculate Mean Average Precision.
        
        Args:
            expected_set: Set of expected tools
            recommended_list: Ordered list of recommendations
            
        Returns:
            MAP score
        """
        if not expected_set:
            return 0.0
        
        num_relevant = 0
        sum_precisions = 0.0
        
        for i, tool_name in enumerate(recommended_list):
            if tool_name in expected_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                sum_precisions += precision_at_i
        
        return sum_precisions / len(expected_set) if expected_set else 0.0
    
    def _calculate_noise_metrics(
        self,
        expected_set: Set[str],
        recommended_list: List[str],
        noise_tool_names: Set[str]
    ) -> Dict[str, float]:
        """
        Calculate noise impact metrics.
        
        Args:
            expected_set: Set of expected tools
            recommended_list: Ordered list of recommendations
            noise_tool_names: Set of noise tool names
            
        Returns:
            Dictionary of noise metrics
        """
        metrics = {}
        
        # Categorize recommended tools
        recommended_noise = [t for t in recommended_list if t in noise_tool_names]
        recommended_expected = [t for t in recommended_list if t in expected_set]
        
        # Noise proportion in results
        noise_proportion = (
            len(recommended_noise) / len(recommended_list)
            if recommended_list else 0.0
        )
        metrics["noise_proportion"] = noise_proportion
        
        # Expected tool recall (critical metric)
        expected_tool_recall = (
            len(recommended_expected) / len(expected_set)
            if expected_set else 1.0
        )
        metrics["expected_tool_recall"] = expected_tool_recall
        
        # Noise resistance score (higher is better)
        # Combines low noise proportion with high expected recall
        noise_resistance = (1 - noise_proportion) * expected_tool_recall
        metrics["noise_resistance"] = noise_resistance
        
        # Additional noise statistics
        metrics["num_noise_in_results"] = len(recommended_noise)
        metrics["num_expected_in_results"] = len(recommended_expected)
        
        # Average rank of expected tools (lower is better)
        if recommended_expected:
            expected_ranks = [
                recommended_list.index(t) + 1
                for t in recommended_expected
            ]
            metrics["avg_rank_expected"] = sum(expected_ranks) / len(expected_ranks)
            metrics["best_rank_expected"] = min(expected_ranks)
        else:
            metrics["avg_rank_expected"] = float('inf')
            metrics["best_rank_expected"] = float('inf')
        
        return metrics
    
    def _empty_ranking_metrics(self) -> Dict[str, float]:
        """Return empty ranking metrics structure."""
        metrics = {
            "mrr": 0.0,
            "map": 0.0,
            "first_relevant_rank": float('inf')
        }
        
        for k in self.config.ndcg_cutoffs:
            metrics[f"ndcg@{k}"] = 0.0
        
        for k in self.config.precision_cutoffs:
            metrics[f"p@{k}"] = 0.0
            metrics[f"r@{k}"] = 0.0
        
        for k in [3, 5, 10]:
            metrics[f"expected_in_top{k}"] = 0
        
        return metrics
    
    def aggregate_metrics(
        self,
        results: List[EvaluationResult],
        noise_tool_names: Optional[Set[str]] = None
    ) -> AggregatedMetrics:
        """
        Aggregate metrics across multiple evaluation results.
        
        Args:
            results: List of evaluation results
            noise_tool_names: Optional set of noise tool names
            
        Returns:
            Aggregated metrics
        """
        if not results:
            return AggregatedMetrics(
                metrics={},
                per_test_metrics=[],
                statistical_summary={},
                metadata={"num_results": 0}
            )
        
        # Calculate metrics for each result
        per_test_metrics = []
        for result in results:
            if result.is_successful:
                metrics = self.calculate_metrics_for_result(result, noise_tool_names)
                per_test_metrics.append(metrics)
        
        if not per_test_metrics:
            return AggregatedMetrics(
                metrics={},
                per_test_metrics=[],
                statistical_summary={},
                metadata={"num_results": len(results), "num_successful": 0}
            )
        
        # Aggregate metrics
        aggregated = {}
        statistical_summary = {}
        
        # Get all metric names
        all_metric_names = set()
        for metrics in per_test_metrics:
            all_metric_names.update(metrics.keys())
        
        # Calculate aggregates for each metric
        for metric_name in all_metric_names:
            values = [
                m.get(metric_name, 0.0)
                for m in per_test_metrics
                if metric_name in m and not math.isinf(m[metric_name])
            ]
            
            if values:
                # Calculate statistics
                aggregated[f"mean_{metric_name}"] = np.mean(values)
                aggregated[f"median_{metric_name}"] = np.median(values)
                aggregated[f"std_{metric_name}"] = np.std(values)
                aggregated[f"min_{metric_name}"] = np.min(values)
                aggregated[f"max_{metric_name}"] = np.max(values)
                
                # Store detailed statistics
                statistical_summary[metric_name] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "p25": np.percentile(values, 25),
                    "p75": np.percentile(values, 75),
                    "p95": np.percentile(values, 95)
                }
        
        # Add success rate
        num_successful = len(per_test_metrics)
        num_total = len(results)
        aggregated["success_rate"] = num_successful / num_total if num_total > 0 else 0.0
        
        return AggregatedMetrics(
            metrics=aggregated,
            per_test_metrics=per_test_metrics,
            statistical_summary=statistical_summary,
            metadata={
                "num_results": num_total,
                "num_successful": num_successful,
                "metric_names": list(all_metric_names)
            }
        )
    
    def calculate_threshold_metrics(
        self,
        results: List[Dict[str, Any]],
        expected_tools: List[Set[str]],
        threshold: float
    ) -> Dict[str, float]:
        """
        Calculate metrics at a specific threshold.
        
        Args:
            results: List of all search results with scores
            expected_tools: List of expected tool sets
            threshold: Score threshold to evaluate
            
        Returns:
            Metrics at the given threshold
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        for i, all_results in enumerate(results):
            expected_set = expected_tools[i] if i < len(expected_tools) else set()
            
            # Filter results by threshold
            filtered = [
                r for r in all_results
                if r.get("score", 0.0) >= threshold
            ]
            
            # Get tool names
            filtered_names = {
                r.get("name", "")
                for r in filtered
                if r.get("name")
            }
            
            all_names = {
                r.get("name", "")
                for r in all_results
                if r.get("name")
            }
            
            # Calculate confusion matrix components
            tp = len(expected_set & filtered_names)
            fp = len(filtered_names - expected_set)
            fn = len(expected_set - filtered_names)
            tn = len((all_names - expected_set) - filtered_names)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn
        
        # Calculate metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
        
        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "true_negatives": total_tn
        }
    
    def find_optimal_threshold(
        self,
        results: List[Dict[str, Any]],
        expected_tools: List[Set[str]],
        metric: str = "f1_score"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find the optimal threshold that maximizes a given metric.
        
        Args:
            results: List of all search results with scores
            expected_tools: List of expected tool sets
            metric: Metric to optimize (default: f1_score)
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        # Collect all unique scores
        all_scores = set()
        for result_list in results:
            for r in result_list:
                score = r.get("score", 0.0)
                if score > 0:
                    all_scores.add(score)
        
        if not all_scores:
            return 0.5, self.calculate_threshold_metrics(results, expected_tools, 0.5)
        
        # Sort scores for efficient search
        sorted_scores = sorted(all_scores)
        
        # Try different thresholds
        best_threshold = 0.5
        best_metric_value = 0.0
        best_metrics = {}
        
        for threshold in sorted_scores:
            metrics = self.calculate_threshold_metrics(results, expected_tools, threshold)
            
            if metrics[metric] > best_metric_value:
                best_metric_value = metrics[metric]
                best_threshold = threshold
                best_metrics = metrics
        
        return best_threshold, best_metrics
    
    def compare_runs(
        self,
        runs: List[EvaluationRun]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics across multiple evaluation runs.
        
        Args:
            runs: List of evaluation runs to compare
            
        Returns:
            Comparison dictionary: run_id -> metric -> value
        """
        comparison = {}
        
        for run in runs:
            # Aggregate metrics for this run
            aggregated = self.aggregate_metrics(run.results)
            comparison[run.id] = aggregated.metrics
        
        return comparison