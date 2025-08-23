"""
Configuration management for evaluation framework.
Single Responsibility: Managing all configuration aspects of evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
from pathlib import Path
from src.services.search_service import SearchStrategy


class TestExecutionMode(Enum):
    """Execution modes for test runner."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    BATCH = "batch"


@dataclass
class SearchStrategyConfig:
    """Configuration for search strategies."""
    strategy: SearchStrategy = SearchStrategy.HYBRID_LTR
    enable_query_enhancement: bool = True
    enable_hybrid_search: bool = True
    enable_cross_encoder: bool = True
    enable_ltr: bool = True
    
    # Thresholds
    primary_similarity_threshold: float = 0.5
    secondary_similarity_threshold: float = 0.3
    two_stage_threshold: int = 100
    
    # Search limits
    max_tools: int = 10
    search_limit: int = 100
    
    # Model configurations
    primary_embedding_model: str = "text-embedding-ada-002"
    secondary_embedding_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy.value,
            "enable_query_enhancement": self.enable_query_enhancement,
            "enable_hybrid_search": self.enable_hybrid_search,
            "enable_cross_encoder": self.enable_cross_encoder,
            "enable_ltr": self.enable_ltr,
            "primary_similarity_threshold": self.primary_similarity_threshold,
            "secondary_similarity_threshold": self.secondary_similarity_threshold,
            "two_stage_threshold": self.two_stage_threshold,
            "max_tools": self.max_tools,
            "search_limit": self.search_limit,
            "primary_embedding_model": self.primary_embedding_model,
            "secondary_embedding_model": self.secondary_embedding_model
        }


@dataclass
class NoiseConfig:
    """Configuration for noise injection in tests."""
    add_noise_to_store: bool = True
    add_noise_to_available: int = 0
    noise_pool_size: int = 1500
    noise_data_source: str = "toolbench"
    noise_data_path: Path = Path("toolbench_data/data/test_instruction")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "add_noise_to_store": self.add_noise_to_store,
            "add_noise_to_available": self.add_noise_to_available,
            "noise_pool_size": self.noise_pool_size,
            "noise_data_source": self.noise_data_source,
            "noise_data_path": str(self.noise_data_path)
        }


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_source: str = "toolbench"
    data_path: Path = Path("toolbench_data/data/test_instruction")
    test_file: str = "G1_instruction.json"
    num_cases: int = 50
    clear_collection_between_tests: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_source": self.data_source,
            "data_path": str(self.data_path),
            "test_file": self.test_file,
            "num_cases": self.num_cases,
            "clear_collection_between_tests": self.clear_collection_between_tests
        }


@dataclass
class ExecutionConfig:
    """Configuration for test execution."""
    mode: TestExecutionMode = TestExecutionMode.SEQUENTIAL
    batch_size: int = 10
    num_workers: int = 4
    timeout_per_test: float = 30.0
    retry_on_failure: bool = False
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "timeout_per_test": self.timeout_per_test,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries
        }


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    calculate_traditional_metrics: bool = True
    calculate_ranking_metrics: bool = True
    calculate_noise_metrics: bool = True
    optimize_threshold: bool = True
    
    # Ranking metric cutoffs
    ndcg_cutoffs: List[int] = field(default_factory=lambda: [3, 5, 10])
    precision_cutoffs: List[int] = field(default_factory=lambda: [1, 3, 5])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "calculate_traditional_metrics": self.calculate_traditional_metrics,
            "calculate_ranking_metrics": self.calculate_ranking_metrics,
            "calculate_noise_metrics": self.calculate_noise_metrics,
            "optimize_threshold": self.optimize_threshold,
            "ndcg_cutoffs": self.ndcg_cutoffs,
            "precision_cutoffs": self.precision_cutoffs
        }


@dataclass
class ReportingConfig:
    """Configuration for reporting."""
    output_dir: Path = Path("evaluation_results")
    save_detailed_results: bool = True
    save_summary: bool = True
    generate_comparison_table: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv", "markdown"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir),
            "save_detailed_results": self.save_detailed_results,
            "save_summary": self.save_summary,
            "generate_comparison_table": self.generate_comparison_table,
            "export_formats": self.export_formats
        }


@dataclass
class EvaluationConfig:
    """
    Main configuration class for evaluation framework.
    Aggregates all configuration aspects.
    """
    
    # Sub-configurations
    search: SearchStrategyConfig = field(default_factory=SearchStrategyConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # Metadata
    experiment_name: str = "default"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EvaluationConfig':
        """Create configuration from dictionary."""
        return cls(
            search=SearchStrategyConfig(**config_dict.get("search", {})),
            noise=NoiseConfig(**config_dict.get("noise", {})),
            data=DataConfig(**config_dict.get("data", {})),
            execution=ExecutionConfig(**config_dict.get("execution", {})),
            metrics=MetricsConfig(**config_dict.get("metrics", {})),
            reporting=ReportingConfig(**config_dict.get("reporting", {})),
            experiment_name=config_dict.get("experiment_name", "default"),
            description=config_dict.get("description", ""),
            tags=config_dict.get("tags", [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "search": self.search.to_dict(),
            "noise": self.noise.to_dict(),
            "data": self.data.to_dict(),
            "execution": self.execution.to_dict(),
            "metrics": self.metrics.to_dict(),
            "reporting": self.reporting.to_dict(),
            "experiment_name": self.experiment_name,
            "description": self.description,
            "tags": self.tags
        }
    
    def save(self, filepath: Path):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'EvaluationConfig':
        """Load configuration from file."""
        import json
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))