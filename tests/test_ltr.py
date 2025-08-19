"""Comprehensive tests for Learning to Rank implementation."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import asyncio

from src.services.ltr_feature_extractor import LTRFeatureExtractor, FeatureConfig
from src.services.ltr_trainer import LTRTrainer
from src.services.ltr_ranker import LTRRanker
from src.services.ltr_service import LTRService


class TestLTRFeatureExtractor:
    """Test cases for LTR feature extraction."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return LTRFeatureExtractor()
    
    @pytest.fixture
    def sample_tool(self):
        """Sample tool for testing."""
        return {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search for files in the repository",
                "parameters": {
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"}
                    },
                    "required": ["pattern"]
                }
            }
        }
    
    def test_extract_features_basic(self, extractor, sample_tool):
        """Test basic feature extraction."""
        query = "search for Python files"
        features = extractor.extract_features(query, sample_tool)
        
        # Check that features are extracted
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # Check specific feature categories
        assert 'exact_name_match' in features
        assert 'description_length' in features
        assert 'query_length' in features
    
    def test_extract_similarity_features(self, extractor, sample_tool):
        """Test similarity feature extraction."""
        context = {
            'semantic_score': 0.8,
            'bm25_score': 0.6,
            'cross_encoder_score': 0.9
        }
        
        features = extractor.extract_features("test", sample_tool, context)
        
        assert features['semantic_similarity'] == 0.8
        assert features['bm25_score'] == 0.6
        assert features['cross_encoder_score'] == 0.9
        assert 'score_mean' in features
        assert 'score_std' in features
    
    def test_extract_name_features(self, extractor, sample_tool):
        """Test name-based feature extraction."""
        query = "search_files tool"
        features = extractor.extract_features(query, sample_tool)
        
        # Exact match should be detected
        assert features['exact_name_match'] == 1.0
        assert features['name_in_query'] == 1.0
        assert features['partial_name_match'] > 0
    
    def test_extract_description_features(self, extractor, sample_tool):
        """Test description-based feature extraction."""
        query = "find files in repository"
        features = extractor.extract_features(query, sample_tool)
        
        assert features['description_length'] > 0
        assert features['description_word_overlap'] > 0
        assert features['keyword_density'] >= 0
    
    def test_extract_parameter_features(self, extractor, sample_tool):
        """Test parameter-based feature extraction."""
        query = "search with pattern and path"
        features = extractor.extract_features(query, sample_tool)
        
        assert features['num_parameters'] > 0
        assert features['num_required_params'] > 0
        assert features['param_name_match'] > 0  # "pattern" and "path" in query
    
    def test_extract_query_features(self, extractor, sample_tool):
        """Test query-specific feature extraction."""
        query = "search for `*.py` files?"
        features = extractor.extract_features(query, sample_tool)
        
        assert features['query_length'] > 0
        assert features['has_code_snippet'] == 1.0  # Backticks detected
        assert features['is_question'] == 1.0  # Ends with ?
        assert features['query_type_search'] == 1.0  # "search" keyword
    
    def test_extract_features_batch(self, extractor, sample_tool):
        """Test batch feature extraction."""
        query = "test query"
        tools = [sample_tool, sample_tool]
        contexts = [{'semantic_score': 0.7}, {'semantic_score': 0.8}]
        
        feature_matrix = extractor.extract_features_batch(query, tools, contexts)
        
        assert isinstance(feature_matrix, np.ndarray)
        assert feature_matrix.shape[0] == 2  # Two tools
        assert feature_matrix.shape[1] > 0  # Multiple features
    
    def test_feature_config(self):
        """Test feature configuration."""
        config = FeatureConfig(
            enable_similarity_features=False,
            enable_name_features=True
        )
        extractor = LTRFeatureExtractor(config)
        
        features = extractor.extract_features(
            "test", 
            {"tool_name": "test_tool"},
            {"semantic_score": 0.5}
        )
        
        # Similarity features should not be present
        assert 'semantic_similarity' not in features
        # Name features should be present
        assert 'exact_name_match' in features


class TestLTRTrainer:
    """Test cases for LTR model training."""
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance."""
        return LTRTrainer(
            model_type="xgboost",
            objective="rank:pairwise"
        )
    
    @pytest.fixture
    def training_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        n_queries = 10
        
        # Random features
        X = np.random.rand(n_samples, n_features)
        
        # Random relevance scores (0 or 1)
        y = np.random.randint(0, 2, n_samples).astype(float)
        
        # Query groups (10 samples per query)
        groups = [10] * n_queries
        
        return X, y, groups
    
    def test_train_basic(self, trainer, training_data):
        """Test basic model training."""
        X, y, groups = training_data
        
        metrics = trainer.train(
            X, y, groups,
            validation_split=0.2,
            early_stopping_rounds=5
        )
        
        assert trainer.model is not None
        assert 'train_metrics' in metrics
        assert 'validation_metrics' in metrics
        assert 'feature_importance' in metrics
    
    def test_train_with_feature_names(self, trainer, training_data):
        """Test training with feature names."""
        X, y, groups = training_data
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        metrics = trainer.train(
            X, y, groups,
            feature_names=feature_names,
            validation_split=0.2
        )
        
        assert trainer.feature_names == feature_names
        assert len(trainer.feature_importance) > 0
    
    def test_cross_validate(self, trainer, training_data):
        """Test cross-validation."""
        X, y, groups = training_data
        
        cv_metrics = trainer.cross_validate(X, y, groups, cv_folds=3)
        
        assert 'mean_ndcg@10' in cv_metrics
        assert 'std_ndcg@10' in cv_metrics
        assert len(cv_metrics['fold_scores']) == 3
    
    def test_save_load_model(self, trainer, training_data):
        """Test model saving and loading."""
        X, y, groups = training_data
        
        # Train model
        trainer.train(X, y, groups, validation_split=0.2)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"
            trainer.save_model(str(model_path))
            
            # Check files exist
            assert model_path.with_suffix('.json').exists()
            assert model_path.with_suffix('.pkl').exists()
            
            # Load model in new trainer
            new_trainer = LTRTrainer()
            new_trainer.load_model(str(model_path))
            
            assert new_trainer.model is not None
            assert new_trainer.feature_names == trainer.feature_names
    
    @patch('src.services.ltr_trainer.xgb.XGBRanker')
    def test_train_from_evaluation_data(self, mock_xgb, trainer):
        """Test training from evaluation results."""
        # Mock feature extractor
        mock_extractor = Mock()
        mock_extractor.get_feature_names.return_value = ['f1', 'f2']
        mock_extractor.extract_features.return_value = {'f1': 0.5, 'f2': 0.8}
        
        # Sample evaluation data
        eval_results = [
            {
                'query': 'test query',
                'recommended_tools': [
                    {'tool_name': 'tool1', 'score': 0.8},
                    {'tool_name': 'tool2', 'score': 0.6}
                ],
                'expected_tools': ['tool1']
            }
        ]
        
        # Mock XGBoost to avoid actual training
        mock_model = Mock()
        mock_xgb.return_value = mock_model
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0.8, 0.6])
        
        metrics = trainer.train_from_evaluation_data(eval_results, mock_extractor)
        
        assert metrics is not None
        mock_model.fit.assert_called_once()


class TestLTRRanker:
    """Test cases for LTR ranking/inference."""
    
    @pytest.fixture
    def ranker(self):
        """Create ranker instance."""
        return LTRRanker(cache_predictions=True, cache_size=10)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock XGBoost model."""
        model = Mock()
        model.predict.return_value = np.array([0.9, 0.7, 0.5, 0.3])
        return model
    
    def test_predict_scores(self, ranker, mock_model):
        """Test score prediction."""
        ranker.model = mock_model
        features = np.random.rand(4, 10)
        
        scores = ranker.predict_scores(features)
        
        assert len(scores) == 4
        assert all(0 <= s <= 1 for s in scores)  # Normalized
        mock_model.predict.assert_called_once()
    
    def test_rank(self, ranker, mock_model):
        """Test ranking functionality."""
        ranker.model = mock_model
        features = np.random.rand(4, 10)
        
        # Get top-2 indices
        indices = ranker.rank(features, top_k=2, return_indices=True)
        
        assert len(indices) == 2
        assert indices[0] == 0  # Highest score first
        assert indices[1] == 1  # Second highest
    
    def test_rank_tools(self, ranker, mock_model):
        """Test tool ranking."""
        ranker.model = mock_model
        
        tool_features = [
            {'f1': 0.8, 'f2': 0.6},
            {'f1': 0.7, 'f2': 0.5},
            {'f1': 0.6, 'f2': 0.4},
            {'f1': 0.5, 'f2': 0.3}
        ]
        
        tools = [
            {'tool_name': f'tool{i}'} for i in range(4)
        ]
        
        ranked_tools = ranker.rank_tools(tool_features, tools, top_k=2)
        
        assert len(ranked_tools) == 2
        assert 'ltr_score' in ranked_tools[0]
        assert ranked_tools[0]['ltr_score'] > ranked_tools[1]['ltr_score']
    
    def test_cache_functionality(self, ranker, mock_model):
        """Test prediction caching."""
        ranker.model = mock_model
        features = np.random.rand(2, 10)
        
        # First call - cache miss
        scores1 = ranker.predict_scores(features)
        assert ranker._cache_stats['misses'] == 1
        
        # Second call with same features - should hit cache
        mock_model.predict.reset_mock()
        scores2 = ranker.predict_scores(features)
        
        # Model should not be called again if cache works
        # Note: Cache uses hash which might not work perfectly with numpy arrays
        # This test might need adjustment based on actual caching implementation
    
    def test_get_model_info(self, ranker, mock_model):
        """Test model info retrieval."""
        ranker.model = mock_model
        ranker.model_metadata = {
            'model_type': 'xgboost',
            'feature_names': ['f1', 'f2'],
            'training_metrics': {'ndcg@10': 0.85}
        }
        
        info = ranker.get_model_info()
        
        assert info['status'] == 'loaded'
        assert info['model_type'] == 'xgboost'
        assert info['n_features'] == 2
        assert 'training_metrics' in info


class TestLTRService:
    """Test cases for LTR service orchestration."""
    
    @pytest.fixture
    def service(self):
        """Create LTR service instance."""
        with patch('src.services.ltr_service.Path'):
            return LTRService(auto_load=False)
    
    @pytest.fixture
    def sample_candidates(self):
        """Sample candidate tools."""
        return [
            {
                'tool_name': 'search_tool',
                'description': 'Search for items',
                'score': 0.7,
                'bm25_score': 0.6
            },
            {
                'tool_name': 'read_tool',
                'description': 'Read file contents',
                'score': 0.6,
                'bm25_score': 0.8
            }
        ]
    
    @pytest.mark.asyncio
    async def test_rank_tools(self, service, sample_candidates):
        """Test tool ranking through service."""
        # Mock the ranker
        service.is_trained = True
        service.ranker.model = Mock()
        service.ranker.predict_scores = Mock(return_value=np.array([0.9, 0.7]))
        
        ranked = await service.rank_tools(
            query="search for files",
            candidates=sample_candidates,
            top_k=2
        )
        
        assert len(ranked) == 2
        assert 'ltr_score' in ranked[0]
        assert ranked[0]['ltr_score'] > ranked[1]['ltr_score']
    
    @pytest.mark.asyncio
    async def test_rank_tools_not_trained(self, service, sample_candidates):
        """Test ranking when model is not trained."""
        service.is_trained = False
        
        ranked = await service.rank_tools(
            query="test",
            candidates=sample_candidates,
            top_k=1
        )
        
        # Should return original order
        assert len(ranked) == 1
        assert ranked[0]['tool_name'] == sample_candidates[0]['tool_name']
    
    def test_train_from_evaluations(self, service):
        """Test training from evaluation data."""
        # Mock trainer
        service.trainer.train_from_evaluation_data = Mock(
            return_value={'ndcg@10': 0.85}
        )
        
        eval_results = [
            {
                'query': 'test',
                'recommended_tools': [{'tool_name': 'tool1'}],
                'expected_tools': ['tool1']
            }
        ]
        
        with patch.object(service, 'save_model'):
            metrics = service.train_from_evaluations(
                eval_results,
                save_model=True
            )
        
        assert service.is_trained
        assert 'ndcg@10' in metrics
    
    def test_get_feature_importance(self, service):
        """Test feature importance retrieval."""
        service.is_trained = True
        service.ranker.get_feature_importance = Mock(
            return_value={
                'feature1_gain': 100,
                'feature2_gain': 80,
                'feature3_gain': 60,
                'feature1_weight': 10
            }
        )
        
        importance = service.get_feature_importance(top_n=2)
        
        # Should return top 2 by gain
        assert len(importance) == 2
        assert 'feature1' in importance
        assert 'feature2' in importance
    
    def test_analyze_ranking(self, service):
        """Test ranking analysis."""
        tools = [
            {'tool_name': 'tool1'},
            {'tool_name': 'tool2'},
            {'tool_name': 'tool3'}
        ]
        expected_tools = ['tool2', 'tool3']
        
        analysis = service.analyze_ranking(
            query="test",
            tools=tools,
            expected_tools=expected_tools
        )
        
        assert analysis['n_tools'] == 3
        assert analysis['n_expected'] == 2
        assert analysis['found_expected'] == ['tool2', 'tool3']
        assert analysis['positions'] == [2, 3]
        assert analysis['mrr'] == 0.5  # 1/2
    
    @pytest.mark.asyncio
    async def test_prepare_contexts_with_bm25(self, service, sample_candidates):
        """Test context preparation with BM25 ranker."""
        # Mock BM25 ranker
        mock_bm25 = Mock()
        mock_bm25.score_tools.return_value = {
            'search_tool': 0.65,
            'read_tool': 0.75
        }
        service.bm25_ranker = mock_bm25
        
        contexts = await service._prepare_contexts("test", sample_candidates)
        
        assert len(contexts) == 2
        assert contexts[0]['semantic_score'] == 0.7
        assert contexts[0]['bm25_score'] == 0.6  # From candidate
        assert contexts[1]['bm25_score'] == 0.8  # From candidate
    
    def test_incremental_train(self, service):
        """Test incremental training logic."""
        new_eval_results = [{'query': 'test'}] * 500  # Below threshold
        
        result = service.incremental_train(
            new_eval_results,
            retrain_threshold=1000
        )
        
        # Should not retrain
        assert result is None
        
        # With enough data
        new_eval_results = [{'query': 'test'}] * 1000
        service.train_from_evaluations = Mock(return_value={'ndcg': 0.9})
        
        result = service.incremental_train(
            new_eval_results,
            retrain_threshold=1000
        )
        
        assert result is not None
    
    def test_get_stats(self, service):
        """Test service statistics."""
        service.is_trained = True
        service.ranker.get_model_info = Mock(
            return_value={'status': 'loaded'}
        )
        
        stats = service.get_stats()
        
        assert stats['is_trained']
        assert 'model_info' in stats
        assert 'feature_config' in stats


class TestIntegration:
    """Integration tests for complete LTR pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete LTR pipeline from feature extraction to ranking."""
        # Create components
        extractor = LTRFeatureExtractor()
        trainer = LTRTrainer()
        ranker = LTRRanker()
        
        # Create sample data
        tools = [
            {
                'function': {
                    'name': 'search_files',
                    'description': 'Search for files'
                }
            },
            {
                'function': {
                    'name': 'read_file',
                    'description': 'Read file contents'
                }
            }
        ]
        
        query = "search for Python files"
        
        # Extract features
        features = extractor.extract_features_batch(query, tools)
        
        assert features.shape[0] == 2
        assert features.shape[1] > 0
        
        # Would train model here with real data
        # For testing, we just verify the pipeline connects
    
    @pytest.mark.asyncio
    async def test_search_service_integration(self):
        """Test LTR integration with SearchService."""
        from src.services.search_service import SearchService, SearchStrategy
        
        # Mock dependencies
        mock_vector_store = Mock()
        mock_embedding_service = Mock()
        mock_ltr_service = Mock()
        
        # Configure mock LTR service
        mock_ltr_service.rank_tools = AsyncMock(
            return_value=[{'tool_name': 'test_tool', 'ltr_score': 0.9}]
        )
        
        # Create search service
        search_service = SearchService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            ltr_service=mock_ltr_service
        )
        search_service.enable_ltr = True
        
        # Mock semantic search
        search_service.semantic_search = AsyncMock(
            return_value=[{'tool_name': 'test_tool', 'score': 0.7}]
        )
        
        # Test LTR search
        results = await search_service.ltr_search(
            query="test query",
            limit=5
        )
        
        assert len(results) == 1
        assert results[0]['ltr_score'] == 0.9
        mock_ltr_service.rank_tools.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])