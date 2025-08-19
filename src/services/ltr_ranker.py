"""Learning to Rank inference/scoring service.

This module is responsible ONLY for scoring and ranking with trained LTR models.
Follows Single Responsibility Principle - only handles inference logic.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import xgboost as xgb
from functools import lru_cache

logger = logging.getLogger(__name__)


class LTRRanker:
    """
    Performs ranking using trained LTR models.
    
    Single Responsibility: Scoring and ranking with trained models.
    Does NOT handle feature extraction or training.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_predictions: bool = True,
        cache_size: int = 1000
    ):
        """
        Initialize LTR ranker.
        
        Args:
            model_path: Path to trained model
            cache_predictions: Whether to cache predictions
            cache_size: Size of prediction cache
        """
        self.model = None
        self.model_metadata = {}
        self.cache_predictions = cache_predictions
        self.cache_size = cache_size
        
        if model_path:
            self.load_model(model_path)
        
        # Initialize cache if enabled
        if self.cache_predictions:
            self._init_cache()
    
    def _init_cache(self):
        """Initialize LRU cache for predictions."""
        
        @lru_cache(maxsize=self.cache_size)
        def _cached_predict(features_hash: str) -> np.ndarray:
            """Cached prediction function."""
            # This will be called by predict_scores
            pass
        
        self._cached_predict = _cached_predict
        self._cache_stats = {'hits': 0, 'misses': 0}
    
    def load_model(self, model_path: str):
        """
        Load trained model from disk.
        
        Args:
            model_path: Path to model files
        """
        path = Path(model_path)
        
        # Load XGBoost model
        model_file = path.with_suffix('.json')
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = xgb.XGBRanker()
        self.model.load_model(str(model_file))
        
        # Load metadata
        metadata_file = path.with_suffix('.pkl')
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.model_metadata = pickle.load(f)
        
        logger.info(f"Loaded LTR model from {model_file}")
        
        # Clear cache on model reload
        if self.cache_predictions:
            self._cached_predict.cache_clear()
            self._cache_stats = {'hits': 0, 'misses': 0}
    
    def predict_scores(
        self,
        features: np.ndarray,
        return_normalized: bool = True
    ) -> np.ndarray:
        """
        Predict relevance scores for samples.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            return_normalized: Whether to normalize scores to [0, 1]
            
        Returns:
            Array of relevance scores
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Try cache if enabled
        if self.cache_predictions:
            features_hash = self._hash_features(features)
            
            # Check if we have cached predictions
            try:
                cached = self._get_cached_predictions(features_hash)
                if cached is not None:
                    self._cache_stats['hits'] += 1
                    return cached
            except:
                pass  # Cache miss or error
            
            self._cache_stats['misses'] += 1
        
        # Predict scores
        scores = self.model.predict(features)
        
        # Normalize if requested
        if return_normalized:
            scores = self._normalize_scores(scores)
        
        # Cache predictions if enabled
        if self.cache_predictions:
            self._store_cached_predictions(features_hash, scores)
        
        return scores
    
    def rank(
        self,
        features: np.ndarray,
        top_k: Optional[int] = None,
        return_indices: bool = False
    ) -> List[int]:
        """
        Rank samples by relevance scores.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            top_k: Return only top-k items
            return_indices: Return indices instead of ranks
            
        Returns:
            List of ranks or indices
        """
        scores = self.predict_scores(features, return_normalized=False)
        
        # Get ranking indices (descending order)
        ranking_indices = np.argsort(-scores)
        
        if top_k is not None:
            ranking_indices = ranking_indices[:top_k]
        
        if return_indices:
            return ranking_indices.tolist()
        else:
            # Convert to ranks (1-indexed)
            ranks = np.empty_like(ranking_indices)
            ranks[ranking_indices] = np.arange(len(ranking_indices)) + 1
            return ranks.tolist()
    
    def rank_tools(
        self,
        tool_features: List[Dict[str, float]],
        tools: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Rank tools based on features.
        
        Args:
            tool_features: List of feature dictionaries
            tools: List of tool objects
            top_k: Number of top tools to return
            
        Returns:
            Ranked list of tools with scores
        """
        if not tool_features or not tools:
            return []
        
        if len(tool_features) != len(tools):
            raise ValueError("Number of features and tools must match")
        
        # Convert features to matrix
        feature_matrix = self._features_to_matrix(tool_features)
        
        # Get scores
        scores = self.predict_scores(feature_matrix)
        
        # Combine tools with scores
        scored_tools = []
        for tool, score in zip(tools, scores):
            # Create result with score
            result = tool.copy() if isinstance(tool, dict) else {"tool": tool}
            result['ltr_score'] = float(score)
            scored_tools.append(result)
        
        # Sort by score
        scored_tools.sort(key=lambda x: x['ltr_score'], reverse=True)
        
        # Return top-k
        return scored_tools[:top_k]
    
    def batch_rank(
        self,
        feature_batches: List[np.ndarray],
        top_k: int = 10
    ) -> List[List[int]]:
        """
        Rank multiple batches of samples.
        
        Args:
            feature_batches: List of feature matrices
            top_k: Number of top items per batch
            
        Returns:
            List of ranking indices for each batch
        """
        rankings = []
        
        for features in feature_batches:
            ranking = self.rank(features, top_k=top_k, return_indices=True)
            rankings.append(ranking)
        
        return rankings
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.
        
        Returns:
            Dictionary of feature names to importance scores
        """
        if self.model is None:
            return {}
        
        # Try to get from metadata first
        if 'feature_importance' in self.model_metadata:
            return self.model_metadata['feature_importance']
        
        # Calculate from model
        try:
            importance = self.model.get_booster().get_score(importance_type='gain')
            
            # Map to feature names if available
            feature_names = self.model_metadata.get('feature_names', [])
            if feature_names:
                mapped_importance = {}
                for key, value in importance.items():
                    if key.startswith('f') and key[1:].isdigit():
                        idx = int(key[1:])
                        if idx < len(feature_names):
                            mapped_importance[feature_names[idx]] = value
                    else:
                        mapped_importance[key] = value
                return mapped_importance
            
            return importance
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}
    
    def calibrate_scores(
        self,
        scores: np.ndarray,
        method: str = "sigmoid"
    ) -> np.ndarray:
        """
        Calibrate scores for better interpretability.
        
        Args:
            scores: Raw scores
            method: Calibration method ('sigmoid', 'isotonic', 'none')
            
        Returns:
            Calibrated scores
        """
        if method == "none":
            return scores
        elif method == "sigmoid":
            # Simple sigmoid calibration
            return 1 / (1 + np.exp(-scores))
        elif method == "isotonic":
            # Would need isotonic regression model
            logger.warning("Isotonic calibration not implemented, using sigmoid")
            return 1 / (1 + np.exp(-scores))
        else:
            return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if self.model is None:
            return {'status': 'no_model_loaded'}
        
        info = {
            'status': 'loaded',
            'model_type': self.model_metadata.get('model_type', 'unknown'),
            'objective': self.model_metadata.get('objective', 'unknown'),
            'n_features': len(self.model_metadata.get('feature_names', [])),
            'feature_names': self.model_metadata.get('feature_names', []),
            'training_metrics': self.model_metadata.get('training_metrics', {}),
        }
        
        if self.cache_predictions:
            info['cache_stats'] = self.get_cache_stats()
        
        return info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        if not self.cache_predictions:
            return {'enabled': False}
        
        stats = {
            'enabled': True,
            'size': self.cache_size,
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'hit_rate': self._cache_stats['hits'] / max(1, self._cache_stats['hits'] + self._cache_stats['misses']),
        }
        
        if hasattr(self, '_cached_predict'):
            stats['current_size'] = self._cached_predict.cache_info().currsize
            stats['max_size'] = self._cached_predict.cache_info().maxsize
        
        return stats
    
    def clear_cache(self):
        """Clear prediction cache."""
        if self.cache_predictions and hasattr(self, '_cached_predict'):
            self._cached_predict.cache_clear()
            self._cache_stats = {'hits': 0, 'misses': 0}
            logger.info("Prediction cache cleared")
    
    # --- Helper methods ---
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores) * 0.5
        
        return (scores - min_score) / (max_score - min_score)
    
    def _features_to_matrix(self, feature_dicts: List[Dict[str, float]]) -> np.ndarray:
        """Convert list of feature dictionaries to numpy matrix."""
        if not feature_dicts:
            return np.array([])
        
        # Get all feature names
        all_features = set()
        for fd in feature_dicts:
            all_features.update(fd.keys())
        
        # Use consistent ordering
        feature_names = sorted(all_features)
        
        # Create matrix
        matrix = np.zeros((len(feature_dicts), len(feature_names)))
        for i, fd in enumerate(feature_dicts):
            for j, fname in enumerate(feature_names):
                matrix[i, j] = fd.get(fname, 0.0)
        
        return matrix
    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash of features for caching."""
        # Use tobytes for consistent hashing
        return str(hash(features.tobytes()))
    
    def _get_cached_predictions(self, features_hash: str) -> Optional[np.ndarray]:
        """Get cached predictions if available."""
        if hasattr(self, '_prediction_cache'):
            return self._prediction_cache.get(features_hash)
        return None
    
    def _store_cached_predictions(self, features_hash: str, predictions: np.ndarray):
        """Store predictions in cache."""
        if not hasattr(self, '_prediction_cache'):
            self._prediction_cache = {}
        
        # Simple size limit
        if len(self._prediction_cache) >= self.cache_size:
            # Remove oldest (simple FIFO)
            first_key = next(iter(self._prediction_cache))
            del self._prediction_cache[first_key]
        
        self._prediction_cache[features_hash] = predictions