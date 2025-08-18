"""
Adaptive Threshold Strategies for Dynamic Score Distributions
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveThreshold:
    """Compute dynamic thresholds based on current score distribution"""
    
    @staticmethod
    def percentile_based(scores: List[float], percentile: float = 80) -> float:
        """
        Use percentile of current scores as threshold.
        E.g., percentile=80 means keep top 20% of tools.
        
        Pros: Adapts to score distribution of current query
        Cons: Always returns same proportion regardless of quality
        """
        if not scores:
            return 0.0
        return np.percentile(scores, 100 - percentile)
    
    @staticmethod
    def statistical_threshold(scores: List[float], method: str = "mean_std") -> float:
        """
        Use statistical properties of current score distribution.
        
        Methods:
        - mean_std: mean - 1*std (includes ~84% of normal distribution)
        - median_mad: median - MAD (robust to outliers)
        - iqr: Q1 - 1.5*IQR (outlier detection method)
        """
        if not scores:
            return 0.0
            
        if method == "mean_std":
            mean = np.mean(scores)
            std = np.std(scores)
            return max(0, mean - std)
            
        elif method == "median_mad":
            median = np.median(scores)
            mad = np.median(np.abs(scores - median))
            return max(0, median - mad)
            
        elif method == "iqr":
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            return max(0, q1 - 1.5 * iqr)
            
        return 0.0
    
    @staticmethod
    def elbow_detection(scores: List[float]) -> float:
        """
        Find the "elbow" in the score distribution where the drop-off accelerates.
        Good for finding natural clustering in scores.
        """
        if len(scores) < 3:
            return min(scores) if scores else 0.0
            
        sorted_scores = sorted(scores, reverse=True)
        
        # Calculate second derivative to find maximum curvature
        max_curvature = 0
        elbow_idx = 0
        
        for i in range(1, len(sorted_scores) - 1):
            curvature = abs(sorted_scores[i-1] - 2*sorted_scores[i] + sorted_scores[i+1])
            if curvature > max_curvature:
                max_curvature = curvature
                elbow_idx = i
                
        return sorted_scores[elbow_idx]
    
    @staticmethod
    def gap_statistic(scores: List[float], min_gap: float = 0.05) -> float:
        """
        Find the largest gap between consecutive scores.
        Tools above the gap are likely relevant, below are likely not.
        """
        if len(scores) < 2:
            return min(scores) if scores else 0.0
            
        sorted_scores = sorted(scores, reverse=True)
        
        max_gap = 0
        gap_threshold = sorted_scores[-1]  # Default to minimum
        
        for i in range(len(sorted_scores) - 1):
            gap = sorted_scores[i] - sorted_scores[i + 1]
            if gap > max_gap and gap >= min_gap:
                max_gap = gap
                gap_threshold = sorted_scores[i + 1]
                
        return gap_threshold
    
    @staticmethod
    def query_aware_threshold(
        scores: List[float],
        query_length: int,
        num_available_tools: int,
        base_threshold: float = 0.4
    ) -> float:
        """
        Adjust threshold based on query characteristics.
        
        - Short queries tend to be less specific -> lower threshold
        - Many available tools -> higher threshold (be more selective)
        - Few available tools -> lower threshold (be more inclusive)
        """
        if not scores:
            return base_threshold
            
        # Adjust based on query length (proxy for specificity)
        query_factor = min(1.2, max(0.8, query_length / 50))  # Normalize around 50 chars
        
        # Adjust based on number of tools (proxy for selection pressure)
        tool_factor = min(1.2, max(0.8, num_available_tools / 10))  # Normalize around 10 tools
        
        # Combine factors
        adjusted_threshold = base_threshold * query_factor * tool_factor
        
        # Ensure threshold is within reasonable bounds based on actual scores
        min_score = min(scores)
        max_score = max(scores)
        
        return max(min_score, min(max_score * 0.9, adjusted_threshold))
    
    @staticmethod
    def hybrid_adaptive(
        scores: List[float],
        semantic_scores: List[float],
        bm25_scores: List[float],
        semantic_weight: float = 0.7
    ) -> float:
        """
        Compute separate thresholds for semantic and BM25, then combine.
        This accounts for different score distributions in each component.
        """
        if not scores:
            return 0.0
            
        # Get thresholds for each component
        semantic_threshold = AdaptiveThreshold.statistical_threshold(semantic_scores)
        bm25_threshold = AdaptiveThreshold.statistical_threshold(bm25_scores)
        
        # Combine thresholds using same weights as score combination
        combined_threshold = (semantic_weight * semantic_threshold + 
                            (1 - semantic_weight) * bm25_threshold)
        
        # Ensure at least some results pass
        if all(s < combined_threshold for s in scores):
            # If nothing passes, use percentile to ensure some results
            return AdaptiveThreshold.percentile_based(scores, percentile=90)
            
        return combined_threshold


class RankBasedSelector:
    """
    Select tools based on ranking rather than absolute threshold.
    This avoids the threshold problem entirely.
    """
    
    @staticmethod
    def top_k_with_diversity(
        scored_tools: List[Dict],
        k: int = 5,
        diversity_penalty: float = 0.1
    ) -> List[Dict]:
        """
        Select top-K tools with diversity bonus.
        Penalizes tools that are too similar to already selected ones.
        """
        if not scored_tools or k <= 0:
            return []
            
        selected = []
        remaining = scored_tools.copy()
        
        while len(selected) < k and remaining:
            # Select highest scoring remaining tool
            best_idx = 0
            best_score = remaining[0]['score']
            
            for i, tool in enumerate(remaining):
                # Apply diversity penalty based on similarity to selected tools
                diversity_score = tool['score']
                
                for selected_tool in selected:
                    # Simple name similarity as proxy for functional similarity
                    name_similarity = _name_similarity(
                        tool['tool_name'], 
                        selected_tool['tool_name']
                    )
                    diversity_score -= diversity_penalty * name_similarity
                    
                if diversity_score > best_score:
                    best_score = diversity_score
                    best_idx = i
                    
            selected.append(remaining.pop(best_idx))
            
        return selected
    
    @staticmethod
    def score_gap_cutoff(
        scored_tools: List[Dict],
        max_k: int = 10,
        min_gap: float = 0.05
    ) -> List[Dict]:
        """
        Return tools until we hit a significant score gap.
        This finds natural clusters in the score distribution.
        """
        if not scored_tools:
            return []
            
        # Sort by score
        sorted_tools = sorted(scored_tools, key=lambda x: x['score'], reverse=True)
        
        selected = [sorted_tools[0]]
        
        for i in range(1, min(len(sorted_tools), max_k)):
            score_gap = sorted_tools[i-1]['score'] - sorted_tools[i]['score']
            
            if score_gap > min_gap:
                # Significant gap found, stop here
                break
                
            selected.append(sorted_tools[i])
            
        return selected
    
    @staticmethod
    def confidence_based(
        scored_tools: List[Dict],
        confidence_threshold: float = 0.8,
        max_k: int = 10
    ) -> List[Dict]:
        """
        Return tools until confidence drops below threshold.
        Confidence = score / max_score (relative scoring).
        """
        if not scored_tools:
            return []
            
        sorted_tools = sorted(scored_tools, key=lambda x: x['score'], reverse=True)
        max_score = sorted_tools[0]['score']
        
        selected = []
        for tool in sorted_tools[:max_k]:
            confidence = tool['score'] / max_score if max_score > 0 else 0
            
            if confidence >= confidence_threshold:
                selected.append(tool)
            else:
                break
                
        return selected


def _name_similarity(name1: str, name2: str) -> float:
    """Simple name similarity based on common tokens"""
    tokens1 = set(name1.lower().split('_'))
    tokens2 = set(name2.lower().split('_'))
    
    if not tokens1 or not tokens2:
        return 0.0
        
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    return len(intersection) / len(union)