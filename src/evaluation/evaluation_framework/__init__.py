"""
Evaluation Framework - A modular, extensible evaluation system.
Follows SOLID principles for clean architecture.
"""

# Configuration imports
from .config import (
    EvaluationConfig,
    SearchStrategyConfig,
    NoiseConfig,
    DataConfig,
    ExecutionConfig,
    MetricsConfig,
    ReportingConfig,
    TestExecutionMode
)

# Data loading imports
from .data_loader import (
    BaseDataLoader,
    ToolBenchDataLoader,
    NoiseDataLoader,
    DataLoaderFactory
)

# Model imports
from .models import (
    TestCase,
    TestSuite,
    EvaluationResult,
    EvaluationRun,
    ComparisonResult,
    TestStatus,
    MetricValue
)

# Test execution imports
from .test_runner import (
    TestRunner,
    TestExecutor,
    SequentialExecutor,
    ParallelExecutor,
    BatchExecutor,
    TestContext
)

# Metrics imports
from .metrics_calculator import (
    MetricsCalculator,
    MetricType,
    MetricResult,
    AggregatedMetrics
)

# Reporting imports
from .reporter import (
    EvaluationReporter,
    ReportFormat,
    BaseReporter,
    JSONReporter,
    CSVReporter,
    MarkdownReporter,
    HTMLReporter
)

# Orchestrator imports
from .orchestrator import (
    EvaluationOrchestrator,
    ExperimentTracker,
    ResultCache
)

# Comparison imports
from .comparison import (
    StrategyComparator,
    StrategyConfig,  # Added for strategy-specific configuration
    MultiConfigRunner
)

__all__ = [
    # Configuration
    'EvaluationConfig',
    'SearchStrategyConfig',
    'NoiseConfig',
    'DataConfig',
    'ExecutionConfig',
    'MetricsConfig',
    'ReportingConfig',
    'TestExecutionMode',
    
    # Data Loading
    'BaseDataLoader',
    'ToolBenchDataLoader',
    'NoiseDataLoader',
    'DataLoaderFactory',
    
    # Models
    'TestCase',
    'TestSuite',
    'EvaluationResult',
    'EvaluationRun',
    'ComparisonResult',
    'TestStatus',
    'MetricValue',
    
    # Test Execution
    'TestRunner',
    'TestExecutor',
    'SequentialExecutor',
    'ParallelExecutor',
    'BatchExecutor',
    'TestContext',
    
    # Metrics
    'MetricsCalculator',
    'MetricType',
    'MetricResult',
    'AggregatedMetrics',
    
    # Reporting
    'EvaluationReporter',
    'ReportFormat',
    'BaseReporter',
    'JSONReporter',
    'CSVReporter',
    'MarkdownReporter',
    'HTMLReporter',
    
    # Orchestration
    'EvaluationOrchestrator',
    'ExperimentTracker',
    'ResultCache',
    
    # Comparison
    'StrategyComparator',
    'StrategyConfig',
    'MultiConfigRunner'
]