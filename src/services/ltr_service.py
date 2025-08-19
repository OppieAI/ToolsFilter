"""Learning to Rank orchestration service.

This module orchestrates the LTR pipeline by coordinating feature extraction,
training, and ranking components. Follows DRY and separation of concerns.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from datetime import datetime
import numpy as np

from src.core.config import get_settings
from src.services.ltr_feature_extractor import LTRFeatureExtractor, FeatureConfig
from src.services.ltr_trainer import LTRTrainer
from src.services.ltr_ranker import LTRRanker
from src.services.bm25_ranker import BM25Ranker
from src.services.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)
settings = get_settings()


class LTRService:
    """
    Main orchestrator for Learning to Rank.
    
    Coordinates feature extraction, training, and ranking components.
    Follows separation of concerns - delegates specific tasks to specialized components.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        bm25_ranker: Optional[BM25Ranker] = None,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        auto_load: bool = True
    ):
        """
        Initialize LTR service.
        
        Args:
            model_path: Path to trained LTR model
            bm25_ranker: Optional BM25 ranker instance (for features)
            cross_encoder: Optional cross-encoder instance (for features)
            auto_load: Whether to auto-load model if exists
        """
        # Initialize components (separation of concerns)
        self.feature_extractor = LTRFeatureExtractor(
            config=self._get_feature_config()
        )
        self.trainer = LTRTrainer(
            model_type="xgboost",
            objective=getattr(settings, 'ltr_objective', 'rank:pairwise'),
            config=self._get_training_config()
        )
        self.ranker = LTRRanker(
            cache_predictions=getattr(settings, 'ltr_cache_predictions', True),
            cache_size=getattr(settings, 'ltr_cache_size', 1000)
        )
        
        # Store references to other rankers (for feature extraction)
        self.bm25_ranker = bm25_ranker
        self.cross_encoder = cross_encoder
        
        # Model management
        self.model_path = model_path or getattr(
            settings, 'ltr_model_path', './models/ltr_xgboost'
        )
        self.is_trained = False
        self.training_history = []
        
        # Auto-load model if exists
        if auto_load and Path(self.model_path).with_suffix('.json').exists():
            self.load_model()
    
    def _get_feature_config(self) -> FeatureConfig:
        """Get feature extraction configuration from settings."""
        return FeatureConfig(
            enable_similarity_features=getattr(settings, 'ltr_similarity_features', True),
            enable_name_features=getattr(settings, 'ltr_name_features', True),
            enable_description_features=getattr(settings, 'ltr_description_features', True),
            enable_parameter_features=getattr(settings, 'ltr_parameter_features', True),
            enable_query_features=getattr(settings, 'ltr_query_features', True),
            enable_metadata_features=getattr(settings, 'ltr_metadata_features', True),
        )
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration from settings."""
        return {
            'learning_rate': getattr(settings, 'ltr_learning_rate', 0.1),
            'max_depth': getattr(settings, 'ltr_max_depth', 6),
            'n_estimators': getattr(settings, 'ltr_n_estimators', 100),
            'subsample': getattr(settings, 'ltr_subsample', 0.8),
            'colsample_bytree': getattr(settings, 'ltr_colsample_bytree', 0.8),
            'min_child_weight': getattr(settings, 'ltr_min_child_weight', 1),
            'gamma': getattr(settings, 'ltr_gamma', 0),
            'reg_alpha': getattr(settings, 'ltr_reg_alpha', 0.0),
            'reg_lambda': getattr(settings, 'ltr_reg_lambda', 1.0),
            'seed': getattr(settings, 'ltr_seed', 42),
            'n_jobs': getattr(settings, 'ltr_n_jobs', -1),
        }
    
    async def rank_tools(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10,
        include_features: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Rank tools using LTR model.
        
        Args:
            query: Search query
            candidates: List of candidate tools
            top_k: Number of top results to return
            include_features: Whether to include features in results
            
        Returns:
            Ranked list of tools with LTR scores
        """
        if not self.is_trained:
            logger.warning("LTR model not trained, returning original order")
            return candidates[:top_k]
        
        if not candidates:
            return []
        
        # Extract features for each candidate
        contexts = await self._prepare_contexts(query, candidates)
        features = self.feature_extractor.extract_features_batch(
            query, candidates, contexts
        )
        
        # Get LTR scores
        scores = self.ranker.predict_scores(features)
        
        # Combine with candidates
        ranked_tools = []
        for i, (tool, score) in enumerate(zip(candidates, scores)):
            result = tool.copy() if isinstance(tool, dict) else {"tool": tool}
            result['ltr_score'] = float(score)
            result['score'] = float(score)  # Use LTR as final score
            
            if include_features:
                result['ltr_features'] = dict(zip(
                    self.feature_extractor.get_feature_names(),
                    features[i].tolist()
                ))
            
            ranked_tools.append(result)
        
        # Sort by LTR score
        ranked_tools.sort(key=lambda x: x['ltr_score'], reverse=True)
        
        # Log ranking changes
        self._log_ranking_changes(candidates, ranked_tools, top_k)
        
        return ranked_tools[:top_k]
    
    async def _prepare_contexts(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare context for feature extraction (includes other model scores).
        
        This method demonstrates DRY principle - reuses existing scorers.
        """
        contexts = []
        
        for candidate in candidates:
            context = {}
            
            # Get existing scores if available
            context['semantic_score'] = candidate.get('semantic_score', 
                                                     candidate.get('score', 0.0))
            
            # Get BM25 score if available
            if self.bm25_ranker and 'bm25_score' not in candidate:
                # Calculate BM25 score if not present
                bm25_scores = self.bm25_ranker.score_tools(query, [candidate])
                tool_name = self._get_tool_name(candidate)
                context['bm25_score'] = bm25_scores.get(tool_name, 0.0)
            else:
                context['bm25_score'] = candidate.get('bm25_score', 0.0)
            
            # Get cross-encoder score if available
            if self.cross_encoder and 'cross_encoder_score' not in candidate:
                # Would need async version of cross_encoder.rerank
                # For now, use existing score if available
                context['cross_encoder_score'] = candidate.get('cross_encoder_score', 0.0)
            else:
                context['cross_encoder_score'] = candidate.get('cross_encoder_score', 0.0)
            
            # Add metadata
            context['category'] = candidate.get('category', 'unknown')
            context['usage_count'] = candidate.get('usage_count', 0)
            
            contexts.append(context)
        
        return contexts
    
    def train_from_evaluations(
        self,
        evaluation_results: List[Dict[str, Any]],
        validation_split: float = 0.2,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train LTR model from evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
            validation_split: Fraction for validation
            save_model: Whether to save trained model
            
        Returns:
            Training metrics
        """
        logger.info(f"Training LTR model from {len(evaluation_results)} evaluations")
        
        # Train model (delegates to trainer component)
        metrics = self.trainer.train_from_evaluation_data(
            evaluation_results,
            self.feature_extractor
        )
        
        # Update ranker with trained model
        self.ranker.model = self.trainer.model
        self.ranker.model_metadata = {
            'feature_names': self.trainer.feature_names,
            'feature_importance': self.trainer.feature_importance,
            'training_metrics': metrics,
        }
        
        self.is_trained = True
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        # Track training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_samples': metrics.get('n_samples', 0),
            'metrics': metrics,
        })
        
        return metrics
    
    def incremental_train(
        self,
        new_evaluation_results: List[Dict[str, Any]],
        retrain_threshold: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Incrementally train model with new evaluation data.
        
        Args:
            new_evaluation_results: New evaluation results
            retrain_threshold: Min new samples before retraining
            
        Returns:
            Training metrics if retrained, None otherwise
        """
        retrain_threshold = retrain_threshold or getattr(
            settings, 'ltr_retrain_threshold', 1000
        )
        
        # Check if we should retrain
        if len(new_evaluation_results) < retrain_threshold:
            logger.info(
                f"Not enough new data for retraining "
                f"({len(new_evaluation_results)} < {retrain_threshold})"
            )
            return None
        
        # Combine with previous training data if available
        # (This would require storing training data, simplified here)
        
        # Retrain
        return self.train_from_evaluations(new_evaluation_results)
    
    def cross_validate(
        self,
        evaluation_results: List[Dict[str, Any]],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on evaluation data.
        
        Args:
            evaluation_results: Evaluation results
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        # Prepare data
        X, y, groups = self.trainer._prepare_training_data(
            evaluation_results,
            self.feature_extractor
        )
        
        # Run cross-validation
        cv_metrics = self.trainer.cross_validate(X, y, groups, cv_folds)
        
        return cv_metrics
    
    def save_model(self, path: Optional[str] = None):
        """
        Save trained model.
        
        Args:
            path: Optional path (uses default if not provided)
        """
        path = path or self.model_path
        self.trainer.save_model(path)
        logger.info(f"LTR model saved to {path}")
    
    def load_model(self, path: Optional[str] = None):
        """
        Load trained model.
        
        Args:
            path: Optional path (uses default if not provided)
        """
        path = path or self.model_path
        
        # Load into both trainer and ranker
        self.trainer.load_model(path)
        self.ranker.load_model(path)
        
        self.is_trained = True
        logger.info(f"LTR model loaded from {path}")
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Feature importance dictionary
        """
        if not self.is_trained:
            return {}
        
        importance = self.ranker.get_feature_importance()
        
        # Filter to gain scores only and sort
        gain_importance = {
            k.replace('_gain', ''): v 
            for k, v in importance.items() 
            if k.endswith('_gain')
        }
        
        # Sort and return top N
        sorted_importance = dict(
            sorted(gain_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        )
        
        return sorted_importance
    
    def analyze_ranking(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        expected_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze ranking performance for a query.
        
        Args:
            query: Search query
            tools: Ranked tools
            expected_tools: Expected tool names
            
        Returns:
            Analysis results
        """
        analysis = {
            'query': query,
            'n_tools': len(tools),
            'n_expected': len(expected_tools),
        }
        
        # Find positions of expected tools
        positions = []
        found_expected = []
        
        for i, tool in enumerate(tools):
            tool_name = self._get_tool_name(tool)
            if tool_name in expected_tools:
                positions.append(i + 1)  # 1-indexed
                found_expected.append(tool_name)
        
        analysis['found_expected'] = found_expected
        analysis['missed_expected'] = [t for t in expected_tools if t not in found_expected]
        analysis['positions'] = positions
        
        # Calculate metrics
        if positions:
            analysis['mrr'] = 1.0 / positions[0]  # Reciprocal rank of first
            analysis['avg_position'] = np.mean(positions)
            analysis['recall@10'] = len([p for p in positions if p <= 10]) / len(expected_tools)
        else:
            analysis['mrr'] = 0.0
            analysis['avg_position'] = float('inf')
            analysis['recall@10'] = 0.0
        
        return analysis
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'is_trained': self.is_trained,
            'model_path': self.model_path,
            'feature_config': vars(self.feature_extractor.config),
            'training_history': self.training_history,
        }
        
        if self.is_trained:
            stats['model_info'] = self.ranker.get_model_info()
            stats['top_features'] = self.get_feature_importance(top_n=10)
        
        return stats
    
    # --- Helper methods ---
    
    def _get_tool_name(self, tool: Dict[str, Any]) -> str:
        """Extract tool name from various formats."""
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("name", "")
        elif "tool_name" in tool:
            return tool["tool_name"]
        elif "name" in tool:
            return tool["name"]
        return ""
    
    def _log_ranking_changes(
        self,
        original: List[Dict[str, Any]],
        ranked: List[Dict[str, Any]],
        top_k: int
    ):
        """Log how LTR changed the ranking."""
        original_names = [self._get_tool_name(t) for t in original[:top_k]]
        ranked_names = [self._get_tool_name(t) for t in ranked[:top_k]]
        
        if original_names != ranked_names:
            logger.debug(f"LTR reranking: {original_names} -> {ranked_names}")