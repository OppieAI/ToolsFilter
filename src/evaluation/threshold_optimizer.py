"""
Threshold Optimizer - Data science methods to find optimal similarity threshold
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ScoreLabel:
    """A score with its ground truth label"""
    score: float
    is_relevant: bool
    tool_name: str
    query_id: Optional[int] = None


class ThresholdOptimizer:
    """Find optimal threshold using various data science methods"""

    def __init__(self):
        self.score_labels: List[ScoreLabel] = []

    def add_score(self, score: float, is_relevant: bool, tool_name: str, query_id: Optional[int] = None):
        """Add a score with its ground truth label"""
        self.score_labels.append(ScoreLabel(score, is_relevant, tool_name, query_id))

    def add_scores_from_evaluation(self, all_scores: List[Dict], expected_tools: List[str], query_id: Optional[int] = None):
        """Add scores from an evaluation run"""
        expected_set = set(tool.lower() for tool in expected_tools)

        # Debug logging
        relevant_scores = []
        irrelevant_scores = []

        for score_dict in all_scores:
            tool_name = score_dict.get('tool_name', '').lower()
            score = score_dict.get('score', 0.0)
            is_relevant = tool_name in expected_set

            if is_relevant:
                relevant_scores.append(score)
            else:
                irrelevant_scores.append(score)

            self.add_score(score, is_relevant, tool_name, query_id)

        # Log score distribution for this query
        if relevant_scores or irrelevant_scores:
            logger.debug(f"Query {query_id}: Relevant scores: {relevant_scores}, Irrelevant scores: {irrelevant_scores}")

    def calculate_metrics_at_threshold(self, threshold: float, use_macro_averaging: bool = True) -> Dict[str, float]:
        """
        Calculate precision, recall, F1 at a specific threshold.
        
        Args:
            threshold: Score threshold for classification
            use_macro_averaging: If True, uses macro-averaging (industry standard for IR).
                                If False, uses micro-averaging (legacy behavior).
        
        Returns:
            Dictionary with metrics
        """
        if use_macro_averaging:
            # Macro-averaging: Calculate metrics per query, then average (IR industry standard)
            from collections import defaultdict
            
            # Group scores by query
            queries = defaultdict(list)
            for sl in self.score_labels:
                queries[sl.query_id].append(sl)
            
            # Calculate metrics for each query
            query_metrics = []
            total_tp = 0
            total_fp = 0
            total_fn = 0
            total_tn = 0
            
            for query_id, query_scores in queries.items():
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                
                for sl in query_scores:
                    predicted_relevant = sl.score >= threshold
                    
                    if predicted_relevant and sl.is_relevant:
                        tp += 1
                    elif predicted_relevant and not sl.is_relevant:
                        fp += 1
                    elif not predicted_relevant and sl.is_relevant:
                        fn += 1
                    else:
                        tn += 1
                
                # Per-query metrics
                q_precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # No predictions = perfect precision
                q_recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # No relevant = perfect recall
                q_f1 = 2 * q_precision * q_recall / (q_precision + q_recall) if (q_precision + q_recall) > 0 else 0
                
                query_metrics.append({
                    'precision': q_precision,
                    'recall': q_recall,
                    'f1': q_f1
                })
                
                # Accumulate for reporting
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
            
            # Macro-average across queries
            avg_precision = np.mean([m['precision'] for m in query_metrics])
            avg_recall = np.mean([m['recall'] for m in query_metrics])
            avg_f1 = np.mean([m['f1'] for m in query_metrics])
            
            # Note: Accuracy is still calculated globally as it's less meaningful per-query
            accuracy = (total_tp + total_tn) / len(self.score_labels) if self.score_labels else 0
            
            return {
                'threshold': threshold,
                'precision': avg_precision,
                'recall': avg_recall,
                'f1': avg_f1,
                'accuracy': accuracy,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn,
                'true_negatives': total_tn,
                'averaging': 'macro'
            }
        else:
            # Micro-averaging: Pool all predictions (legacy behavior)
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0

            for sl in self.score_labels:
                predicted_relevant = sl.score >= threshold

                if predicted_relevant and sl.is_relevant:
                    true_positives += 1
                elif predicted_relevant and not sl.is_relevant:
                    false_positives += 1
                elif not predicted_relevant and sl.is_relevant:
                    false_negatives += 1
                else:
                    true_negatives += 1

            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / len(self.score_labels) if self.score_labels else 0

            return {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'true_negatives': true_negatives,
                'averaging': 'micro'
            }

    def grid_search_optimal_threshold(self, step: float = 0.01, metric: str = 'f1') -> Dict[str, float]:
        """
        Grid search to find optimal threshold

        Args:
            step: Step size for threshold search
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy', 'balanced')
        """
        if not self.score_labels:
            return {'threshold': 0.5, 'message': 'No scores available'}

        scores = [sl.score for sl in self.score_labels]
        min_score = min(scores)
        max_score = max(scores)

        best_threshold = min_score
        best_metric_value = 0
        best_metrics = None

        # Try thresholds from min to max score
        threshold = min_score
        all_results = []

        while threshold <= max_score + step:
            metrics = self.calculate_metrics_at_threshold(threshold, use_macro_averaging=True)
            all_results.append(metrics)

            # Determine which metric to optimize
            if metric == 'f1':
                metric_value = metrics['f1']
            elif metric == 'precision':
                metric_value = metrics['precision']
            elif metric == 'recall':
                metric_value = metrics['recall']
            elif metric == 'accuracy':
                metric_value = metrics['accuracy']
            elif metric == 'balanced':
                # Balance between precision and recall
                metric_value = (metrics['precision'] + metrics['recall']) / 2
            else:
                metric_value = metrics['f1']

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold
                best_metrics = metrics

            threshold += step

        return {
            'optimal_threshold': best_threshold,
            'optimized_metric': metric,
            'metrics_at_optimal': best_metrics,
            'all_thresholds': all_results
        }

    def percentile_based_threshold(self, percentile: float = 75) -> float:
        """
        Set threshold based on score percentile

        Args:
            percentile: Percentile to use (0-100)
        """
        if not self.score_labels:
            return 0.5

        scores = [sl.score for sl in self.score_labels]
        return np.percentile(scores, percentile)

    def statistical_threshold(self, method: str = 'mean_std') -> float:
        """
        Calculate threshold using statistical methods

        Args:
            method: 'mean', 'median', 'mean_std', 'median_mad'
        """
        if not self.score_labels:
            return 0.5

        # Separate relevant and irrelevant scores
        relevant_scores = [sl.score for sl in self.score_labels if sl.is_relevant]
        irrelevant_scores = [sl.score for sl in self.score_labels if not sl.is_relevant]

        if method == 'mean':
            # Use mean of relevant scores
            return np.mean(relevant_scores) if relevant_scores else 0.5

        elif method == 'median':
            # Use median of relevant scores
            return np.median(relevant_scores) if relevant_scores else 0.5

        elif method == 'mean_std':
            # Use mean - 1 std of relevant scores
            if relevant_scores:
                return np.mean(relevant_scores) - np.std(relevant_scores)
            return 0.5

        elif method == 'median_mad':
            # Use median - MAD of relevant scores
            if relevant_scores:
                median = np.median(relevant_scores)
                mad = np.median(np.abs(relevant_scores - median))
                return median - mad
            return 0.5

        elif method == 'intersection':
            # Find intersection point between relevant and irrelevant distributions
            if relevant_scores and irrelevant_scores:
                # Simple approach: use midpoint between max(irrelevant) and min(relevant)
                max_irrelevant = max(irrelevant_scores) if irrelevant_scores else 0
                min_relevant = min(relevant_scores) if relevant_scores else 1
                return (max_irrelevant + min_relevant) / 2
            return 0.5

        return 0.5

    def elbow_method_threshold(self) -> float:
        """
        Find threshold using elbow method on score distribution
        """
        if not self.score_labels:
            return 0.5

        # Sort scores
        scores = sorted([sl.score for sl in self.score_labels], reverse=True)

        # Find the elbow point (maximum curvature)
        n = len(scores)
        if n < 3:
            return scores[n//2] if scores else 0.5

        # Calculate second derivative to find maximum curvature
        max_curvature = 0
        elbow_idx = 0

        for i in range(1, n-1):
            # Approximate second derivative
            curvature = abs(scores[i-1] - 2*scores[i] + scores[i+1])
            if curvature > max_curvature:
                max_curvature = curvature
                elbow_idx = i

        return scores[elbow_idx]

    def roc_analysis_threshold(self) -> Dict[str, float]:
        """
        Find optimal threshold using ROC curve analysis
        Maximizes Youden's J statistic (TPR - FPR)
        """
        if not self.score_labels:
            return {'threshold': 0.5, 'auc': 0.5}

        # Calculate TPR and FPR at different thresholds
        scores = sorted(set([sl.score for sl in self.score_labels]))

        best_threshold = scores[0]
        best_j_statistic = -1

        tpr_values = []
        fpr_values = []

        for threshold in scores:
            metrics = self.calculate_metrics_at_threshold(threshold, use_macro_averaging=True)

            # Calculate TPR (True Positive Rate) = Recall
            tpr = metrics['recall']

            # Calculate FPR (False Positive Rate)
            fp = metrics['false_positives']
            tn = metrics['true_negatives']
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_values.append(tpr)
            fpr_values.append(fpr)

            # Youden's J statistic
            j_statistic = tpr - fpr

            if j_statistic > best_j_statistic:
                best_j_statistic = j_statistic
                best_threshold = threshold

        # Calculate AUC (Area Under Curve) using trapezoidal rule
        auc = 0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2

        return {
            'optimal_threshold': best_threshold,
            'auc': auc,
            'j_statistic': best_j_statistic,
            'method': 'roc_analysis'
        }

    def find_optimal_threshold(self, methods: Optional[List[str]] = None) -> Dict[str, any]:
        """
        Find optimal threshold using multiple methods and return recommendations

        Args:
            methods: List of methods to use. If None, uses all methods.
        """
        if methods is None:
            methods = ['grid_search_f1', 'grid_search_balanced', 'roc_analysis',
                      'percentile_75', 'statistical_mean_std', 'elbow']

        results = {}

        for method in methods:
            if method == 'grid_search_f1':
                result = self.grid_search_optimal_threshold(metric='f1')
                results['grid_search_f1'] = result.get('optimal_threshold', result.get('threshold', 0.5))

            elif method == 'grid_search_balanced':
                result = self.grid_search_optimal_threshold(metric='balanced')
                results['grid_search_balanced'] = result.get('optimal_threshold', result.get('threshold', 0.5))

            elif method == 'roc_analysis':
                result = self.roc_analysis_threshold()
                results['roc_analysis'] = result.get('optimal_threshold', result.get('threshold', 0.5))

            elif method == 'percentile_75':
                results['percentile_75'] = self.percentile_based_threshold(75)

            elif method == 'percentile_80':
                results['percentile_80'] = self.percentile_based_threshold(80)

            elif method == 'statistical_mean_std':
                results['statistical_mean_std'] = self.statistical_threshold('mean_std')

            elif method == 'statistical_intersection':
                results['statistical_intersection'] = self.statistical_threshold('intersection')

            elif method == 'elbow':
                results['elbow'] = self.elbow_method_threshold()

        # Calculate consensus (median of all methods)
        all_thresholds = list(results.values())
        consensus_threshold = np.median(all_thresholds) if all_thresholds else 0.5

        # Evaluate consensus threshold using macro-averaging (IR standard)
        consensus_metrics = self.calculate_metrics_at_threshold(consensus_threshold, use_macro_averaging=True)

        return {
            'methods': results,
            'consensus_threshold': consensus_threshold,
            'consensus_metrics': consensus_metrics,
            'score_distribution': self.get_score_distribution()
        }

    def get_score_distribution(self) -> Dict[str, any]:
        """Get statistics about score distribution"""
        if not self.score_labels:
            return {}

        all_scores = [sl.score for sl in self.score_labels]
        relevant_scores = [sl.score for sl in self.score_labels if sl.is_relevant]
        irrelevant_scores = [sl.score for sl in self.score_labels if not sl.is_relevant]

        return {
            'total_scores': len(all_scores),
            'relevant_count': len(relevant_scores),
            'irrelevant_count': len(irrelevant_scores),
            'all': {
                'min': min(all_scores),
                'max': max(all_scores),
                'mean': np.mean(all_scores),
                'median': np.median(all_scores),
                'std': np.std(all_scores)
            },
            'relevant': {
                'min': min(relevant_scores) if relevant_scores else None,
                'max': max(relevant_scores) if relevant_scores else None,
                'mean': np.mean(relevant_scores) if relevant_scores else None,
                'median': np.median(relevant_scores) if relevant_scores else None,
                'std': np.std(relevant_scores) if relevant_scores else None
            },
            'irrelevant': {
                'min': min(irrelevant_scores) if irrelevant_scores else None,
                'max': max(irrelevant_scores) if irrelevant_scores else None,
                'mean': np.mean(irrelevant_scores) if irrelevant_scores else None,
                'median': np.median(irrelevant_scores) if irrelevant_scores else None,
                'std': np.std(irrelevant_scores) if irrelevant_scores else None
            }
        }

    def save_analysis(self, filepath: str):
        """Save threshold analysis to file"""
        analysis = self.find_optimal_threshold()

        # Add detailed grid search results
        grid_search = self.grid_search_optimal_threshold(step=0.01)
        analysis['detailed_grid_search'] = grid_search

        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=float)

    def plot_analysis(self) -> str:
        """Generate a text-based plot of threshold vs metrics"""
        if not self.score_labels:
            return "No data to plot"

        # Get grid search results
        grid_results = self.grid_search_optimal_threshold(step=0.02)

        plot_lines = []
        plot_lines.append("Threshold Analysis")
        plot_lines.append("=" * 60)

        # Create text-based plot
        for result in grid_results['all_thresholds']:
            threshold = result['threshold']
            precision = result['precision']
            recall = result['recall']
            f1 = result['f1']

            # Create bar representation
            p_bar = '*' * int(precision * 20)
            r_bar = '+' * int(recall * 20)
            f_bar = '=' * int(f1 * 20)

            plot_lines.append(f"T:{threshold:.2f} | P:{p_bar:20s} | R:{r_bar:20s} | F1:{f_bar:20s}")

        plot_lines.append("=" * 60)
        plot_lines.append("Legend: P=Precision(*), R=Recall(+), F1=F1-Score(=)")

        return '\n'.join(plot_lines)
