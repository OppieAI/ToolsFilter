"""
Search Pipeline Configuration System

Provides a flexible, extensible configuration system for search pipelines
that eliminates method signature pollution and ensures consistent behavior
across different search use cases.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class PipelineStage(Enum):
    """Pipeline stages in execution order"""
    SEMANTIC = "semantic"
    MULTI_CRITERIA = "multi_criteria" 
    BM25 = "bm25"
    CROSS_ENCODER = "cross_encoder"
    LTR = "ltr"
    POST_PROCESSING = "post_processing"


@dataclass
class SearchPipelineConfig:
    """
    Base pipeline configuration class.
    
    Provides comprehensive control over search pipeline execution with
    sensible defaults and extensibility for specialized use cases.
    """
    
    # =================================================================
    # PIPELINE STAGE CONTROL
    # =================================================================
    enable_semantic: bool = True
    enable_multi_criteria: bool = True
    enable_bm25: bool = True  
    enable_cross_encoder: bool = True
    enable_ltr: bool = True
    enable_post_processing: bool = True
    
    # Stop pipeline execution after specific stage
    stop_after_stage: Optional[PipelineStage] = None
    
    # =================================================================
    # THRESHOLDS AND LIMITS  
    # =================================================================
    semantic_threshold: float = 0.0
    bm25_threshold: float = 0.0
    cross_encoder_threshold: float = 0.0
    final_threshold: float = 0.13  # Applied at the end
    
    # Candidate limits at different stages
    semantic_limit: int = 100
    multi_criteria_limit: int = 100
    cross_encoder_limit: int = 50
    final_limit: int = 10
    
    # =================================================================
    # SCORING CONFIGURATION
    # =================================================================
    # Semantic + BM25 hybrid weights
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    bm25_significance_threshold: float = 0.5  # When to apply hybrid scoring
    
    # Multi-criteria search settings
    exact_match_boost: bool = True
    enable_param_matching: bool = True
    enable_description_matching: bool = True
    
    # =================================================================
    # FEATURE EXTRACTION & DEBUGGING
    # =================================================================
    extract_features: bool = False  # For LTR training
    include_scores: bool = True  # Include individual scores in results
    include_match_types: bool = False  # Include match type info
    debug_mode: bool = False  # Extra logging
    
    # =================================================================
    # CONFIDENCE AND FILTERING
    # =================================================================
    enable_confidence_cutoff: bool = False
    confidence_cutoff_ratio: float = 0.6  # Stop when score < ratio * top_score
    confidence_min_threshold: float = 0.15  # Absolute minimum confidence
    confidence_max_gap: float = 0.1  # Max allowed gap between consecutive results
    
    # =================================================================
    # PERFORMANCE OPTIMIZATIONS
    # =================================================================
    enable_caching: bool = True
    parallel_scoring: bool = False  # For multiple scoring methods
    batch_size: int = 50  # For batch operations
    
    # =================================================================
    # EXTENSIBILITY
    # =================================================================
    custom_params: Dict[str, Any] = field(default_factory=dict)  # For custom extensions
    
    def should_stop_after(self, stage: PipelineStage) -> bool:
        """Check if pipeline should stop after given stage"""
        return self.stop_after_stage == stage
    
    def is_stage_enabled(self, stage: PipelineStage) -> bool:
        """Check if a specific pipeline stage is enabled"""
        stage_mapping = {
            PipelineStage.SEMANTIC: self.enable_semantic,
            PipelineStage.MULTI_CRITERIA: self.enable_multi_criteria,
            PipelineStage.BM25: self.enable_bm25,
            PipelineStage.CROSS_ENCODER: self.enable_cross_encoder,
            PipelineStage.LTR: self.enable_ltr,
            PipelineStage.POST_PROCESSING: self.enable_post_processing,
        }
        return stage_mapping.get(stage, False)
    
    def get_limit_for_stage(self, stage: PipelineStage) -> int:
        """Get candidate limit for specific stage"""
        stage_limits = {
            PipelineStage.SEMANTIC: self.semantic_limit,
            PipelineStage.MULTI_CRITERIA: self.multi_criteria_limit,
            PipelineStage.CROSS_ENCODER: self.cross_encoder_limit,
            PipelineStage.LTR: self.final_limit,
            PipelineStage.POST_PROCESSING: self.final_limit,
        }
        return stage_limits.get(stage, self.final_limit)


@dataclass 
class TrainingPipelineConfig(SearchPipelineConfig):
    """
    Configuration optimized for LTR training.
    
    Stops before LTR stage and enables feature extraction.
    Uses production-identical pipeline for feature generation.
    """
    
    # Stop before LTR to get training features
    enable_ltr: bool = False
    stop_after_stage: PipelineStage = PipelineStage.CROSS_ENCODER
    
    # Enable feature extraction for training
    extract_features: bool = True
    include_scores: bool = True  # Need all scores for features
    include_match_types: bool = True  # Need match type info
    
    # Use production-like limits but get more candidates for training variety
    multi_criteria_limit: int = 200  # More candidates for diverse training
    cross_encoder_limit: int = 100   # More for cross-encoder training
    
    # No final filtering - we want all candidates for training
    final_threshold: float = 0.0
    enable_confidence_cutoff: bool = False
    
    # Enable debugging for training analysis
    debug_mode: bool = True
    

@dataclass
class EvaluationPipelineConfig(SearchPipelineConfig):
    """
    Configuration for evaluation/testing.
    
    Uses production settings with evaluation-specific adjustments.
    """
    
    # Standard evaluation thresholds
    final_threshold: float = 0.13
    final_limit: int = 10
    
    # Enable all production features
    enable_ltr: bool = True
    enable_confidence_cutoff: bool = False  # Let evaluator handle cutoffs
    
    # Include extra info for evaluation analysis
    include_scores: bool = True
    include_match_types: bool = True


@dataclass  
class TwoStagePipelineConfig(SearchPipelineConfig):
    """
    Configuration for two-stage filtering strategy.
    
    Implements aggressive first stage + precise second stage filtering.
    """
    
    # Two-stage specific settings
    stage1_threshold: float = 0.10  # Lower threshold for stage 1
    stage1_limit: int = 50          # Higher limit for stage 1
    stage2_threshold: float = 0.15  # Stricter threshold for stage 2
    
    # Override base settings for two-stage approach
    final_threshold: float = 0.15   # Use stage2_threshold
    enable_confidence_cutoff: bool = True  # Apply confidence cutoffs
    
    # Enable all stages for comprehensive two-stage filtering
    enable_multi_criteria: bool = True
    enable_bm25: bool = True
    enable_cross_encoder: bool = True
    enable_ltr: bool = True


@dataclass
class ProductionPipelineConfig(SearchPipelineConfig):
    """
    Configuration for production use.
    
    Optimized for performance and quality with conservative settings.
    """
    
    # Production-optimized limits
    multi_criteria_limit: int = 100
    cross_encoder_limit: int = 30
    final_limit: int = 10
    
    # Production thresholds
    final_threshold: float = 0.13
    
    # Performance optimizations
    enable_caching: bool = True
    parallel_scoring: bool = False  # Conservative for stability
    
    # Quality features
    enable_confidence_cutoff: bool = True
    exact_match_boost: bool = True
    
    # Minimal debugging in production
    debug_mode: bool = False
    include_match_types: bool = False  # Reduce response size


@dataclass
class DebuggingPipelineConfig(SearchPipelineConfig):
    """
    Configuration for debugging and development.
    
    Maximizes information output and enables detailed logging.
    """
    
    # Enable all debugging features
    debug_mode: bool = True
    include_scores: bool = True
    include_match_types: bool = True
    extract_features: bool = True  # For analysis
    
    # Looser thresholds for debugging
    final_threshold: float = 0.0
    enable_confidence_cutoff: bool = False
    
    # Get more results for analysis
    final_limit: int = 50
    cross_encoder_limit: int = 100
    

@dataclass
class FastPipelineConfig(SearchPipelineConfig):
    """
    Configuration optimized for speed.
    
    Disables expensive operations while maintaining reasonable quality.
    """
    
    # Disable expensive stages
    enable_cross_encoder: bool = False
    enable_ltr: bool = False
    
    # Use only fast stages
    enable_multi_criteria: bool = True
    enable_bm25: bool = True
    stop_after_stage: PipelineStage = PipelineStage.BM25
    
    # Aggressive limits for speed
    multi_criteria_limit: int = 20
    final_limit: int = 5
    
    # Higher threshold for fewer results
    final_threshold: float = 0.20
    
    # Performance optimizations
    enable_caching: bool = True
    parallel_scoring: bool = True


# =================================================================
# FACTORY FUNCTIONS FOR COMMON USE CASES
# =================================================================

def get_training_config(**overrides) -> TrainingPipelineConfig:
    """Get training configuration with optional overrides"""
    config = TrainingPipelineConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_evaluation_config(**overrides) -> EvaluationPipelineConfig:
    """Get evaluation configuration with optional overrides"""
    config = EvaluationPipelineConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_production_config(**overrides) -> ProductionPipelineConfig:
    """Get production configuration with optional overrides"""
    config = ProductionPipelineConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config


def get_two_stage_config(**overrides) -> TwoStagePipelineConfig:
    """Get two-stage filtering configuration with optional overrides"""
    config = TwoStagePipelineConfig()
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
    return config