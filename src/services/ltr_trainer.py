"""Learning to Rank model training service.

This module is responsible ONLY for training the LTR model.
Follows Single Responsibility Principle - only handles model training logic.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import ndcg_score

logger = logging.getLogger(__name__)


class LTRTrainer:
    """
    Trains Learning to Rank models.
    
    Single Responsibility: Training LTR models from evaluation data.
    Does NOT handle feature extraction, scoring, or ranking.
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        objective: str = "rank:pairwise",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LTR trainer.
        
        Args:
            model_type: Type of model to train (currently only 'xgboost')
            objective: Ranking objective ('rank:pairwise', 'rank:ndcg', 'rank:map')
            config: Model configuration parameters
        """
        self.model_type = model_type
        self.objective = objective
        self.config = config or self._get_default_config()
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default XGBoost configuration."""
        return {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'seed': 42,
            'n_jobs': -1,
        }
    
    def train(
        self,
        features: np.ndarray,
        relevance_scores: np.ndarray,
        query_groups: List[int],
        feature_names: Optional[List[str]] = None,
        validation_split: float = 0.2,
        early_stopping_rounds: int = 10
    ) -> Dict[str, Any]:
        """
        Train the LTR model.
        
        Args:
            features: Feature matrix (n_samples x n_features)
            relevance_scores: Relevance scores for each sample
            query_groups: Group sizes for each query (for ranking)
            feature_names: Optional feature names
            validation_split: Fraction of data for validation
            early_stopping_rounds: Rounds for early stopping
            
        Returns:
            Training metrics and statistics
        """
        if self.model_type != "xgboost":
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Training LTR model with {len(features)} samples, {features.shape[1]} features")
        
        self.feature_names = feature_names or [f"f{i}" for i in range(features.shape[1])]
        
        # Prepare data for XGBoost ranker
        X_train, X_val, y_train, y_val, groups_train, groups_val = self._split_data(
            features, relevance_scores, query_groups, validation_split
        )
        
        # Create XGBoost ranker
        self.model = xgb.XGBRanker(
            objective=self.objective,
            learning_rate=self.config['learning_rate'],
            max_depth=self.config['max_depth'],
            n_estimators=self.config['n_estimators'],
            subsample=self.config['subsample'],
            colsample_bytree=self.config['colsample_bytree'],
            min_child_weight=self.config['min_child_weight'],
            gamma=self.config['gamma'],
            reg_alpha=self.config['reg_alpha'],
            reg_lambda=self.config['reg_lambda'],
            random_state=self.config['seed'],
            n_jobs=self.config['n_jobs'],
            early_stopping_rounds=early_stopping_rounds if early_stopping_rounds else None,
        )
        
        # Train model
        logger.info("Starting model training...")
        self.model.fit(
            X_train, y_train,
            group=groups_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            eval_group=[groups_val] if groups_val is not None else None,
            verbose=False
        )
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance()
        
        # Evaluate model
        train_metrics = self._evaluate(X_train, y_train, groups_train)
        val_metrics = self._evaluate(X_val, y_val, groups_val)
        
        # Store training metrics
        self.training_metrics = {
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'feature_importance': self.feature_importance,
            'model_params': self.config,
            'training_date': datetime.now().isoformat(),
            'n_samples': len(features),
            'n_features': features.shape[1],
            'n_queries': len(query_groups),
        }
        
        logger.info(f"Training completed. Validation NDCG@10: {val_metrics['ndcg@10']:.4f}")
        
        return self.training_metrics
    
    def train_from_evaluation_data(
        self,
        evaluation_results: List[Dict[str, Any]],
        feature_extractor: Any
    ) -> Dict[str, Any]:
        """
        Train from ToolBench evaluation results.
        
        Args:
            evaluation_results: List of evaluation results
            feature_extractor: Feature extractor instance
            
        Returns:
            Training metrics
        """
        logger.info(f"Preparing training data from {len(evaluation_results)} evaluations")
        
        # Convert evaluation data to training format
        X, y, groups = self._prepare_training_data(evaluation_results, feature_extractor)
        
        # Get feature names from extractor
        feature_names = feature_extractor.get_feature_names()
        
        # Train model
        return self.train(X, y, groups, feature_names)
    
    def cross_validate(
        self,
        features: np.ndarray,
        relevance_scores: np.ndarray,
        query_groups: List[int],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            features: Feature matrix
            relevance_scores: Relevance scores
            query_groups: Query groups
            cv_folds: Number of CV folds
            
        Returns:
            Cross-validation metrics
        """
        logger.info(f"Running {cv_folds}-fold cross-validation")
        
        # Convert groups to format for sklearn
        group_labels = []
        for i, size in enumerate(query_groups):
            group_labels.extend([i] * size)
        group_labels = np.array(group_labels)
        
        # Create cross-validator
        cv = GroupKFold(n_splits=cv_folds)
        
        # Create model for CV
        model = xgb.XGBRanker(
            objective=self.objective,
            **self.config
        )
        
        # Run cross-validation
        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(features, relevance_scores, group_labels)):
            X_train = features[train_idx]
            y_train = relevance_scores[train_idx]
            X_val = features[val_idx]
            y_val = relevance_scores[val_idx]
            
            # Get groups for this fold
            train_groups = self._get_groups_from_indices(train_idx, group_labels)
            val_groups = self._get_groups_from_indices(val_idx, group_labels)
            
            # Train on fold
            model.fit(X_train, y_train, group=train_groups, verbose=False)
            
            # Evaluate
            fold_metrics = self._evaluate(X_val, y_val, val_groups)
            scores.append(fold_metrics)
            logger.info(f"Fold {fold+1}: NDCG@10={fold_metrics['ndcg@10']:.4f}")
        
        # Aggregate scores
        cv_metrics = {
            'mean_ndcg@10': np.mean([s['ndcg@10'] for s in scores]),
            'std_ndcg@10': np.std([s['ndcg@10'] for s in scores]),
            'mean_ndcg@5': np.mean([s['ndcg@5'] for s in scores]),
            'std_ndcg@5': np.std([s['ndcg@5'] for s in scores]),
            'fold_scores': scores,
        }
        
        logger.info(f"CV completed. Mean NDCG@10: {cv_metrics['mean_ndcg@10']:.4f} ± {cv_metrics['std_ndcg@10']:.4f}")
        
        return cv_metrics
    
    def save_model(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path.with_suffix('.json')
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'objective': self.objective,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'training_metrics': self.training_metrics,
            'config': self.config,
        }
        
        metadata_path = path.with_suffix('.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {model_path} and {metadata_path}")
    
    def load_model(self, path: str):
        """
        Load trained model from disk.
        
        Args:
            path: Path to model files
        """
        path = Path(path)
        
        # Load XGBoost model
        model_path = path.with_suffix('.json')
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = xgb.XGBRanker()
        self.model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = path.with_suffix('.pkl')
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.model_type = metadata.get('model_type', 'xgboost')
            self.objective = metadata.get('objective', 'rank:pairwise')
            self.feature_names = metadata.get('feature_names')
            self.feature_importance = metadata.get('feature_importance')
            self.training_metrics = metadata.get('training_metrics', {})
            self.config = metadata.get('config', {})
        
        logger.info(f"Model loaded from {model_path}")
    
    # --- Helper methods ---
    
    def _split_data(
        self,
        features: np.ndarray,
        relevance_scores: np.ndarray,
        query_groups: List[int],
        validation_split: float
    ) -> Tuple:
        """Split data into train and validation sets."""
        n_queries = len(query_groups)
        n_train_queries = int(n_queries * (1 - validation_split))
        
        # Split by queries
        train_groups = query_groups[:n_train_queries]
        val_groups = query_groups[n_train_queries:]
        
        # Calculate sample indices
        train_size = sum(train_groups)
        
        X_train = features[:train_size]
        X_val = features[train_size:]
        y_train = relevance_scores[:train_size]
        y_val = relevance_scores[train_size:]
        
        return X_train, X_val, y_train, y_val, train_groups, val_groups
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate and return feature importance."""
        if self.model is None:
            return {}
        
        importance_dict = {}
        
        # Get importance scores
        importance_types = ['weight', 'gain', 'cover']
        for imp_type in importance_types:
            scores = self.model.get_booster().get_score(importance_type=imp_type)
            
            # Map to feature names
            for fname, score in scores.items():
                # XGBoost uses f0, f1, etc. if no feature names provided
                if fname.startswith('f') and fname[1:].isdigit():
                    idx = int(fname[1:])
                    if self.feature_names and idx < len(self.feature_names):
                        feature_name = self.feature_names[idx]
                    else:
                        feature_name = fname
                else:
                    feature_name = fname
                
                key = f"{feature_name}_{imp_type}"
                importance_dict[key] = float(score)
        
        return importance_dict
    
    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: List[int]
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if self.model is None:
            return {}
        
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics per query
        ndcg_scores = []
        current_idx = 0
        
        for group_size in groups:
            group_y = y[current_idx:current_idx + group_size]
            group_pred = y_pred[current_idx:current_idx + group_size]
            
            if len(group_y) > 1:
                # Calculate NDCG for this query
                ndcg_10 = self._calculate_ndcg(group_y, group_pred, k=10)
                ndcg_5 = self._calculate_ndcg(group_y, group_pred, k=5)
                ndcg_scores.append({'ndcg@10': ndcg_10, 'ndcg@5': ndcg_5})
            
            current_idx += group_size
        
        # Average metrics
        metrics = {
            'ndcg@10': np.mean([s['ndcg@10'] for s in ndcg_scores]),
            'ndcg@5': np.mean([s['ndcg@5'] for s in ndcg_scores]),
            'n_queries': len(groups),
        }
        
        return metrics
    
    def _calculate_ndcg(self, y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
        """Calculate NDCG@k for a single query."""
        # Ensure we have the right shape
        y_true = np.asarray([y_true]).reshape(1, -1)
        y_pred = np.asarray([y_pred]).reshape(1, -1)
        
        return ndcg_score(y_true, y_pred, k=k)
    
    def _prepare_training_data(
        self,
        evaluation_results: List[Dict[str, Any]],
        feature_extractor: Any
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Convert evaluation results to training data.
        
        Returns:
            features, relevance_scores, query_groups
        """
        all_features = []
        all_relevance = []
        query_groups = []
        
        for eval_result in evaluation_results:
            query = eval_result.get('query', '')
            recommended_tools = eval_result.get('recommended_tools', [])
            expected_tools = eval_result.get('expected_tools', [])
            
            if not recommended_tools:
                continue
            
            # Extract features for each tool
            query_features = []
            query_relevance = []
            
            for tool in recommended_tools:
                # Get tool name
                tool_name = tool.get('tool_name', tool.get('name', ''))
                
                # Determine relevance (1 if in expected, 0 otherwise)
                relevance = 1.0 if tool_name in expected_tools else 0.0
                
                # Extract features
                context = {
                    'semantic_score': tool.get('score', 0.0),
                    'bm25_score': tool.get('bm25_score', 0.0),
                    'cross_encoder_score': tool.get('cross_encoder_score', 0.0),
                }
                
                features = feature_extractor.extract_features(query, tool, context)
                
                # Convert to array
                feature_values = list(features.values())
                query_features.append(feature_values)
                query_relevance.append(relevance)
            
            if query_features:
                all_features.extend(query_features)
                all_relevance.extend(query_relevance)
                query_groups.append(len(query_features))
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_relevance)
        
        return X, y, query_groups
    
    def _get_groups_from_indices(
        self,
        indices: np.ndarray,
        group_labels: np.ndarray
    ) -> List[int]:
        """Get group sizes from indices."""
        selected_groups = group_labels[indices]
        unique_groups = np.unique(selected_groups)
        
        groups = []
        for g in unique_groups:
            groups.append(np.sum(selected_groups == g))
        
        return groups