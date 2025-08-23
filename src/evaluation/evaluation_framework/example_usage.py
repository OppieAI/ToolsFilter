"""
Example usage of the evaluation framework.
Demonstrates various features and capabilities.
"""

import asyncio
from pathlib import Path
from typing import List

# Import framework components
from src.evaluation.evaluation_framework import (
    EvaluationOrchestrator,
    EvaluationConfig,
    SearchStrategyConfig,
    NoiseConfig,
    DataConfig,
    ExecutionConfig,
    MetricsConfig,
    ReportingConfig,
    TestExecutionMode
)
from src.services.search_service import SearchStrategy


async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with default settings.
    """
    print("\n" + "="*60)
    print("Example 1: Basic Evaluation")
    print("="*60)
    
    # Create configuration
    config = EvaluationConfig(
        experiment_name="basic_evaluation",
        description="Basic evaluation with default settings",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=10  # Small number for quick demo
        )
    )
    
    # Create orchestrator and run
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    
    # Get aggregated metrics
    metrics = orchestrator.get_aggregated_metrics()
    print(f"\nMean F1 Score: {metrics.metrics.get('mean_f1_score', 0):.3f}")
    print(f"Mean MRR: {metrics.metrics.get('mean_mrr', 0):.3f}")
    
    return run


async def example_with_noise():
    """
    Example 2: Evaluation with noise injection.
    """
    print("\n" + "="*60)
    print("Example 2: Evaluation with Noise")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="evaluation_with_noise",
        description="Testing noise resistance",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=10
        ),
        noise=NoiseConfig(
            add_noise_to_store=True,
            add_noise_to_available=50,  # Add 50 noise tools per test
            noise_pool_size=500
        ),
        search=SearchStrategyConfig(
            strategy=SearchStrategy.HYBRID_LTR,
            primary_similarity_threshold=0.6
        )
    )
    
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    
    # Check noise impact
    metrics = orchestrator.get_aggregated_metrics()
    print(f"\nNoise Resistance: {metrics.metrics.get('mean_noise_resistance', 0):.3f}")
    print(f"Expected Tool Recall: {metrics.metrics.get('mean_expected_tool_recall', 0):.3f}")
    
    return run


async def example_parallel_execution():
    """
    Example 3: Parallel execution for faster evaluation.
    """
    print("\n" + "="*60)
    print("Example 3: Parallel Execution")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="parallel_evaluation",
        description="Using parallel execution for speed",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=20
        ),
        execution=ExecutionConfig(
            mode=TestExecutionMode.PARALLEL,
            num_workers=4,
            timeout_per_test=30.0,
            retry_on_failure=True,
            max_retries=2
        )
    )
    
    # Track progress
    def progress_callback(completed: int, total: int):
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run(progress_callback=progress_callback)
    
    print(f"\nCompleted in: {run.duration_seconds:.2f} seconds")
    
    return run


async def example_strategy_comparison():
    """
    Example 4: Compare multiple search strategies efficiently.
    Uses StrategyComparator to test all strategies on the same indexed data.
    """
    print("\n" + "="*60)
    print("Example 4: Strategy Comparison (Efficient)")
    print("="*60)
    
    from src.evaluation.evaluation_framework import StrategyComparator
    
    # Base configuration
    config = EvaluationConfig(
        experiment_name="strategy_comparison",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=10
        ),
        noise=NoiseConfig(
            add_noise_to_available=30
        ),
        reporting=ReportingConfig(
            export_formats=["json", "markdown", "html"],
            generate_comparison_table=True
        )
    )
    
    # Strategies to compare
    strategies = [
        SearchStrategy.SEMANTIC,
        SearchStrategy.HYBRID,
        SearchStrategy.HYBRID_LTR
    ]
    
    # Use StrategyComparator for efficient comparison
    comparator = StrategyComparator(config)
    
    # Define progress callback
    def progress_callback(strategy_name: str, completed: int, total: int):
        print(f"  {strategy_name}: {completed}/{total} tests completed")
    
    # Compare strategies (indexes data once, tests all strategies)
    comparison = await comparator.compare_strategies(
        strategies,
        progress_callback=progress_callback,
        parallel_strategies=False  # Set to True for parallel strategy execution
    )
    
    print("\nComparison complete! Best performers identified.")
    
    return comparison


async def example_threshold_optimization():
    """
    Example 5: Find optimal similarity threshold.
    """
    print("\n" + "="*60)
    print("Example 5: Threshold Optimization")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="threshold_optimization",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=20
        ),
        metrics=MetricsConfig(
            optimize_threshold=True
        )
    )
    
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    
    # Find optimal threshold
    optimal_threshold, metrics_at_threshold = orchestrator.find_optimal_threshold(
        metric="f1_score"
    )
    
    print(f"\nCurrent threshold: {config.search.primary_similarity_threshold}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print(f"F1 at optimal: {metrics_at_threshold.get('f1_score', 0):.3f}")
    
    return run


async def example_custom_reporting():
    """
    Example 6: Custom reporting with multiple formats.
    """
    print("\n" + "="*60)
    print("Example 6: Custom Reporting")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="custom_reporting",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=10
        ),
        reporting=ReportingConfig(
            output_dir=Path("custom_reports"),
            export_formats=["json", "csv", "markdown", "html"],
            save_detailed_results=True,
            generate_comparison_table=True
        )
    )
    
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    
    # Export in specific formats
    from src.evaluation.evaluation_framework.reporter import ReportFormat
    
    exported = await orchestrator.export_results(
        formats=[ReportFormat.HTML, ReportFormat.MARKDOWN]
    )
    
    print("\nExported reports:")
    for format, path in exported.items():
        print(f"  {format.value}: {path}")
    
    return run


async def example_with_caching():
    """
    Example 7: Using caching for faster re-runs.
    """
    print("\n" + "="*60)
    print("Example 7: Caching Example")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="cached_evaluation",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=5
        )
    )
    
    # First run - will compute and cache
    print("\nFirst run (computing)...")
    orchestrator = EvaluationOrchestrator(config, use_cache=True)
    run1 = await orchestrator.run()
    
    # Second run - will use cache
    print("\nSecond run (using cache)...")
    orchestrator2 = EvaluationOrchestrator(config, use_cache=True)
    run2 = await orchestrator2.run()
    
    print(f"\nRuns are identical: {run1.id == run2.id}")
    
    return run1


async def example_batch_execution():
    """
    Example 8: Batch execution for balanced performance.
    """
    print("\n" + "="*60)
    print("Example 8: Batch Execution")
    print("="*60)
    
    config = EvaluationConfig(
        experiment_name="batch_execution",
        data=DataConfig(
            test_file="G1_instruction.json",
            num_cases=20
        ),
        execution=ExecutionConfig(
            mode=TestExecutionMode.BATCH,
            batch_size=5,
            timeout_per_test=30.0
        )
    )
    
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    
    print(f"\nProcessed {len(run.results)} tests in batches of {config.execution.batch_size}")
    
    return run


async def main():
    """
    Run all examples.
    """
    print("\n" + "="*80)
    print("EVALUATION FRAMEWORK EXAMPLES")
    print("="*80)
    
    # Choose which examples to run
    examples_to_run = [
        example_basic_evaluation,
        example_with_noise,
        # example_parallel_execution,  # Comment out for quick demo
        # example_strategy_comparison,  # Takes longer
        # example_threshold_optimization,
        example_custom_reporting,
        example_with_caching,
        # example_batch_execution
    ]
    
    results = []
    for example_func in examples_to_run:
        try:
            result = await example_func()
            results.append(result)
        except Exception as e:
            print(f"\nError in {example_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())