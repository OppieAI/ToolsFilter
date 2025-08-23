# Evaluation Framework

## Overview

A modular, extensible evaluation framework for testing and comparing tool filtering/search strategies. Built following SOLID principles and clean architecture patterns to ensure maintainability, testability, and extensibility.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    EvaluationOrchestrator                        │
│  (Main coordinator - orchestrates the entire evaluation flow)    │
└─────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│Configuration │       │ DataLoader   │       │TestRunner    │
│  Manager     │       │              │       │              │
├──────────────┤       ├──────────────┤       ├──────────────┤
│-Search Config│       │-Load datasets│       │-Execute tests│
│-Noise Config │       │-Transform    │       │-Parallel/Seq │
│-Data Config  │       │-Validate     │       │-Retry logic  │
│-Metrics      │       │-Cache        │       │-Timeout      │
└──────────────┘       └──────────────┘       └──────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   MetricsCalculator    │
                    ├────────────────────────┤
                    │-Traditional metrics    │
                    │-Ranking metrics        │
                    │-Noise impact analysis  │
                    │-Threshold optimization │
                    └────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  EvaluationReporter    │
                    ├────────────────────────┤
                    │-Generate reports       │
                    │-Compare runs           │
                    │-Export (JSON/CSV/MD)   │
                    │-Visualizations         │
                    └────────────────────────┘
```

## Core Design Principles

### 1. Single Responsibility Principle (SRP)
Each class has one clear responsibility:
- **EvaluationConfig**: Manages configuration
- **DataLoader**: Handles data loading and transformation
- **TestRunner**: Executes tests
- **MetricsCalculator**: Calculates metrics
- **EvaluationReporter**: Generates reports

### 2. Open/Closed Principle (OCP)
- New data sources can be added by extending `BaseDataLoader`
- New metrics can be added without modifying `MetricsCalculator`
- New report formats can be added by extending `BaseReporter`

### 3. Liskov Substitution Principle (LSP)
- Any `DataLoader` implementation can be used interchangeably
- Any `SearchStrategy` can be substituted without changing the framework

### 4. Interface Segregation Principle (ISP)
- Interfaces are focused and minimal
- Components depend only on the interfaces they need

### 5. Dependency Inversion Principle (DIP)
- High-level modules don't depend on low-level modules
- Both depend on abstractions (interfaces/protocols)

## Components

### 1. Configuration Management (`config.py`)

**Purpose**: Centralized configuration management with validation and serialization.

**Key Classes**:
- `EvaluationConfig`: Main configuration aggregator
- `SearchStrategyConfig`: Search-specific settings
- `NoiseConfig`: Noise injection settings
- `DataConfig`: Data source settings
- `ExecutionConfig`: Test execution settings
- `MetricsConfig`: Metrics calculation settings
- `ReportingConfig`: Output and reporting settings

**Features**:
- Type-safe configuration with dataclasses
- Configuration validation
- JSON serialization/deserialization
- Configuration inheritance and composition
- Environment variable support

### 2. Data Loading (`data_loader.py`)

**Purpose**: Abstract data loading with support for multiple data sources.

**Key Classes**:
- `BaseDataLoader`: Abstract base class defining the interface
- `ToolBenchDataLoader`: Implementation for ToolBench datasets
- `TestCase`: Data model for test cases
- `TestSuite`: Collection of test cases with metadata

**Features**:
- Lazy loading with caching
- Data validation and transformation
- Support for multiple data formats
- Streaming for large datasets
- Built-in data augmentation (noise injection)

### 3. Models (`models.py`)

**Purpose**: Data models for test cases and results.

**Key Classes**:
- `TestCase`: Individual test case representation
- `TestSuite`: Collection of test cases
- `EvaluationResult`: Result of a single test
- `EvaluationRun`: Complete evaluation run with all results
- `ComparisonResult`: Comparison between multiple runs

**Features**:
- Immutable data models
- Serialization support
- Validation
- Rich comparison methods

### 4. Test Execution (`test_runner.py`)

**Purpose**: Execute tests with different strategies (sequential, parallel, batch).

**Key Classes**:
- `TestRunner`: Main test execution engine
- `TestExecutor`: Interface for different execution strategies
- `SequentialExecutor`: Run tests one by one
- `ParallelExecutor`: Run tests in parallel
- `BatchExecutor`: Run tests in batches

**Features**:
- Multiple execution modes
- Progress tracking
- Error handling and retry logic
- Timeout management
- Resource cleanup
- Test isolation

### 5. Metrics Calculation (`metrics_calculator.py`)

**Purpose**: Calculate all evaluation metrics in a centralized, efficient manner.

**Key Classes**:
- `MetricsCalculator`: Main metrics computation engine
- `MetricType`: Enum of available metrics
- `MetricResult`: Individual metric result
- `AggregatedMetrics`: Aggregated metrics across tests

**Metrics Categories**:
- **Traditional**: Precision, Recall, F1
- **Ranking**: MRR, NDCG@k, P@k
- **Noise Impact**: Expected tool recall, noise proportion
- **Performance**: Latency, throughput
- **Statistical**: Mean, std, percentiles

### 6. Reporting (`reporter.py`)

**Purpose**: Generate reports and visualizations from evaluation results.

**Key Classes**:
- `EvaluationReporter`: Main reporting engine
- `ReportFormat`: Enum of output formats
- `ComparisonTable`: Compare multiple evaluation runs
- `VisualizationGenerator`: Create charts and plots

**Features**:
- Multiple export formats (JSON, CSV, Markdown, HTML)
- Comparison tables
- Statistical analysis
- Trend analysis
- Performance dashboards

### 7. Orchestrator (`orchestrator.py`)

**Purpose**: Main coordinator that ties all components together.

**Key Features**:
- High-level API for running evaluations
- Component lifecycle management
- Result aggregation
- Multi-run comparisons
- Experiment tracking

## Data Flow

```
1. Configuration Loading
   ├── Load config from file/dict/env
   └── Validate configuration

2. Data Preparation
   ├── Load test data via DataLoader
   ├── Apply transformations
   └── Inject noise if configured

3. Test Execution
   ├── Initialize services (VectorStore, Embeddings, Search)
   ├── Execute tests via TestRunner
   ├── Handle retries and timeouts
   └── Collect raw results

4. Metrics Calculation
   ├── Calculate per-test metrics
   ├── Aggregate across tests
   ├── Optimize thresholds
   └── Statistical analysis

5. Reporting
   ├── Generate summary reports
   ├── Create comparison tables
   ├── Export to configured formats
   └── Save artifacts
```

## Usage Examples

### Basic Usage

```python
from evaluation_framework import (
    EvaluationOrchestrator,
    EvaluationConfig,
    SearchStrategyConfig
)
from src.services.search_service import SearchStrategy

# Create configuration
config = EvaluationConfig(
    experiment_name="baseline_hybrid_ltr",
    search=SearchStrategyConfig(
        strategy=SearchStrategy.HYBRID_LTR,
        primary_similarity_threshold=0.5
    ),
    data=DataConfig(
        test_file="G1_instruction.json",
        num_cases=50
    ),
    noise=NoiseConfig(
        add_noise_to_available=100
    )
)

# Run evaluation
orchestrator = EvaluationOrchestrator(config)
results = await orchestrator.run()

# Get summary
print(results.summary())
```

### Comparing Multiple Strategies (Efficient)

```python
from evaluation_framework import StrategyComparator, EvaluationConfig

# Base configuration (strategy will be overridden)
config = EvaluationConfig(
    data=DataConfig(test_file="G1_instruction.json", num_cases=50),
    noise=NoiseConfig(add_noise_to_available=100),
    reporting=ReportingConfig(generate_comparison_table=True)
)

# Create comparator
comparator = StrategyComparator(config)

# Define strategies to compare
strategies = [
    SearchStrategy.SEMANTIC,
    SearchStrategy.HYBRID,
    SearchStrategy.HYBRID_CROSS_ENCODER,
    SearchStrategy.HYBRID_LTR
]

# Compare strategies efficiently (indexes data once, tests all strategies)
comparison = await comparator.compare_strategies(
    strategies,
    parallel_strategies=True  # Run strategies in parallel for speed
)

# Results include detailed comparison metrics and best performers
print(comparison.generate_summary())
```

**Key Benefits of StrategyComparator:**
- **Efficient**: Indexes data once, tests all strategies on same data
- **Fair**: Ensures identical test conditions for all strategies
- **Fast**: Option to run strategies in parallel
- **Comprehensive**: Automatic comparison metrics and reports

### Custom Data Loader

```python
from evaluation_framework import BaseDataLoader, TestCase

class CustomDataLoader(BaseDataLoader):
    def load_test_cases(self) -> List[TestCase]:
        # Custom loading logic
        pass
    
    def transform_test_case(self, raw_data: Dict) -> TestCase:
        # Custom transformation
        pass

# Use custom loader
config = EvaluationConfig(
    data=DataConfig(data_source="custom")
)
orchestrator = EvaluationOrchestrator(
    config,
    data_loader=CustomDataLoader()
)
```

### Parallel Execution

```python
config = EvaluationConfig(
    execution=ExecutionConfig(
        mode=TestExecutionMode.PARALLEL,
        num_workers=8,
        timeout_per_test=30.0
    )
)
```

## Extension Points

### Adding New Data Sources

1. Extend `BaseDataLoader`
2. Implement required methods
3. Register in `DataLoaderFactory`

### Adding New Metrics

1. Add metric type to `MetricType` enum
2. Implement calculation in `MetricsCalculator`
3. Update aggregation logic

### Adding New Report Formats

1. Add format to `ReportFormat` enum
2. Implement exporter in `EvaluationReporter`
3. Register format handler

## Configuration Examples

### Minimal Configuration

```json
{
  "experiment_name": "quick_test",
  "data": {
    "test_file": "test.json",
    "num_cases": 10
  }
}
```

### Full Configuration

```json
{
  "experiment_name": "comprehensive_evaluation",
  "description": "Full evaluation with noise and multiple metrics",
  "tags": ["production", "v2.0"],
  "search": {
    "strategy": "HYBRID_LTR",
    "enable_query_enhancement": true,
    "primary_similarity_threshold": 0.5,
    "max_tools": 10
  },
  "noise": {
    "add_noise_to_store": true,
    "add_noise_to_available": 100,
    "noise_pool_size": 1500
  },
  "data": {
    "data_source": "toolbench",
    "test_file": "G1_instruction.json",
    "num_cases": 50
  },
  "execution": {
    "mode": "parallel",
    "num_workers": 4,
    "timeout_per_test": 30.0,
    "retry_on_failure": true
  },
  "metrics": {
    "calculate_traditional_metrics": true,
    "calculate_ranking_metrics": true,
    "optimize_threshold": true,
    "ndcg_cutoffs": [3, 5, 10],
    "precision_cutoffs": [1, 3, 5]
  },
  "reporting": {
    "output_dir": "evaluation_results",
    "export_formats": ["json", "csv", "markdown", "html"],
    "generate_comparison_table": true
  }
}
```

## Migration from Old Evaluator

### Key Changes

1. **Configuration**: Centralized in `EvaluationConfig` instead of scattered
2. **Data Loading**: Abstracted with `DataLoader` interface
3. **Metrics**: Separated into dedicated `MetricsCalculator`
4. **Reporting**: Enhanced with `EvaluationReporter`
5. **Execution**: Flexible with multiple modes via `TestRunner`

### Migration Steps

```python
# Old way
evaluator = ToolBenchEvaluator()
results = await evaluator.run_evaluation(
    test_file="G1_instruction.json",
    num_cases=50,
    add_noise_to_available=100
)

# New way
config = EvaluationConfig(
    data=DataConfig(
        test_file="G1_instruction.json",
        num_cases=50
    ),
    noise=NoiseConfig(
        add_noise_to_available=100
    )
)
orchestrator = EvaluationOrchestrator(config)
results = await orchestrator.run()
```

## TODO List

### Phase 1: Core Components ✅
- [x] Create module structure
- [x] Design configuration system (`config.py`)
- [x] Create comprehensive documentation
- [x] Implement data models (`models.py`)
- [x] Implement base data loader (`data_loader.py`)
- [x] Implement ToolBench data loader

### Phase 2: Execution Engine ✅
- [x] Implement test runner base class
- [x] Implement sequential executor
- [x] Implement parallel executor
- [x] Implement batch executor
- [x] Add retry and timeout logic
- [x] Add progress tracking

### Phase 3: Metrics System ✅
- [x] Implement metrics calculator
- [x] Add traditional metrics (precision, recall, F1)
- [x] Add ranking metrics (MRR, NDCG, P@k)
- [x] Add noise impact metrics
- [x] Add threshold optimization
- [x] Add statistical aggregation

### Phase 4: Reporting ✅
- [x] Implement base reporter
- [x] Add JSON export
- [x] Add CSV export
- [x] Add Markdown export
- [x] Add HTML export with visualizations
- [x] Add comparison table generator

### Phase 5: Orchestration ✅
- [x] Implement main orchestrator
- [x] Add component lifecycle management
- [x] Add experiment tracking
- [x] Add multi-run comparison
- [x] Add caching layer

### Phase 6: Advanced Features
- [ ] Add A/B testing support
- [ ] Add statistical significance testing
- [ ] Add performance profiling
- [ ] Add distributed execution support
- [ ] Add real-time monitoring dashboard
- [ ] Add automatic hyperparameter tuning

### Phase 7: Testing & Documentation
- [ ] Write unit tests for each component
- [ ] Write integration tests
- [ ] Add performance benchmarks
- [ ] Create usage examples
- [ ] Write API documentation
- [ ] Create tutorial notebooks

### Phase 8: Migration
- [ ] Create migration script
- [ ] Update existing tests
- [ ] Deprecate old evaluator
- [ ] Update documentation

## Development Guidelines

### Code Style
- Use type hints for all functions
- Follow PEP 8
- Write docstrings for all public methods
- Keep functions small and focused

### Testing
- Write unit tests for each component
- Aim for >80% code coverage
- Use mocks for external dependencies
- Test edge cases and error conditions

### Performance
- Use async/await for I/O operations
- Batch operations where possible
- Implement caching for expensive operations
- Profile and optimize bottlenecks

## Contributing

### Adding a New Feature
1. Create a design document
2. Get design approval
3. Implement with tests
4. Update documentation
5. Submit PR with clear description

### Reporting Issues
- Include configuration used
- Provide minimal reproduction case
- Include error messages and logs
- Describe expected vs actual behavior

## License

[Your License Here]

## Contact

[Your Contact Information]