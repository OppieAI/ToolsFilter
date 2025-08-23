"""
Migration script from old ToolBenchEvaluator to new evaluation framework.
This script demonstrates how to migrate existing evaluation code.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

# New framework imports
from src.evaluation.evaluation_framework import (
    EvaluationOrchestrator,
    EvaluationConfig,
    SearchStrategyConfig,
    NoiseConfig,
    DataConfig,
    ExecutionConfig,
    MetricsConfig,
    ReportingConfig,
    TestExecutionMode,
    StrategyComparator,  # For efficient strategy comparison
    StrategyConfig       # For strategy-specific configuration
)
from src.services.search_service import SearchStrategy
from src.core.config import get_settings

logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see NDCG debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
settings = get_settings()

def migrate_config(
    test_file: str = "G1_instruction.json",
    data_dir: str = "toolbench_data/data/test_instruction",
    num_cases: int = 50,
    clear_collection: bool = True,
    add_noise_tools: bool = True,
    add_noise_to_available: int = 0
) -> EvaluationConfig:
    """
    Migrate old evaluator parameters to new configuration.

    Old API:
        evaluator = ToolBenchEvaluator()
        results = await evaluator.run_evaluation(
            test_file="G1_instruction.json",
            data_dir="toolbench_data/data/test_instruction",
            num_cases=50,
            clear_collection=True,
            add_noise_tools=True,
            add_noise_to_available=100
        )

    New API:
        config = migrate_config(...)
        orchestrator = EvaluationOrchestrator(config)
        results = await orchestrator.run()
    """

    # Get settings for defaults
    settings = get_settings()

    # Create new configuration
    config = EvaluationConfig(
        experiment_name=f"migrated_{test_file.replace('.json', '')}",
        description="Migrated from old ToolBenchEvaluator",

        # Data configuration
        data=DataConfig(
            data_source="toolbench",
            data_path=Path(data_dir),
            test_file=test_file,
            num_cases=num_cases,
            clear_collection_between_tests=clear_collection
        ),

        # Noise configuration
        noise=NoiseConfig(
            add_noise_to_store=add_noise_tools,
            add_noise_to_available=add_noise_to_available,
            noise_pool_size=1500 if add_noise_tools else 0,
            noise_data_path=Path(data_dir)  # Use the same data directory for noise
        ),

        # Search configuration
        search=SearchStrategyConfig(
            # Determine strategy based on settings
            strategy=SearchStrategy.HYBRID_LTR if settings.enable_ltr else
                    SearchStrategy.HYBRID_CROSS_ENCODER if settings.enable_cross_encoder else
                    SearchStrategy.HYBRID if settings.enable_hybrid_search else
                    SearchStrategy.SEMANTIC,
            enable_query_enhancement=settings.enable_query_enhancement,
            enable_hybrid_search=settings.enable_hybrid_search,
            enable_cross_encoder=settings.enable_cross_encoder,
            enable_ltr=settings.enable_ltr,
            primary_similarity_threshold=settings.primary_similarity_threshold,
            two_stage_threshold=settings.two_stage_threshold,
            max_tools=10,
            primary_embedding_model=settings.primary_embedding_model
        ),

        # Execution configuration
        execution=ExecutionConfig(
            mode=TestExecutionMode.SEQUENTIAL,  # Old evaluator was sequential
            timeout_per_test=30.0,
            retry_on_failure=False
        ),

        # Metrics configuration
        metrics=MetricsConfig(
            calculate_traditional_metrics=True,
            calculate_ranking_metrics=True,
            calculate_noise_metrics=add_noise_to_available > 0,
            optimize_threshold=True,
            ndcg_cutoffs=[3, 5, 10],
            precision_cutoffs=[1, 3, 5]
        ),

        # Reporting configuration
        reporting=ReportingConfig(
            output_dir=Path("evaluation_results"),
            save_detailed_results=True,
            save_summary=True,
            generate_comparison_table=False,
            export_formats=["json", "csv", "markdown"]
        )
    )

    return config


async def migrate_and_run():
    """
    Example of migrating and running with the new framework.
    """
    print("Migration Example: Converting old evaluator code to new framework")
    print("="*60)

    # Old way (for reference - don't actually run this)
    """
    from src.evaluation.toolbench_evaluator import ToolBenchEvaluator

    evaluator = ToolBenchEvaluator()
    results = await evaluator.run_evaluation(
        test_file="G1_instruction.json",
        num_cases=20,
        clear_collection=True,
        add_noise_tools=True,
        add_noise_to_available=100
    )
    """

    # New way - using migration helper
    print("\n1. Migrating configuration...")
    config = migrate_config(
        test_file="G1_instruction.json",
        num_cases=20,
        clear_collection=True,
        add_noise_tools=True,
        add_noise_to_available=100
    )

    print("\n2. Creating orchestrator...")
    orchestrator = EvaluationOrchestrator(config)

    print("\n3. Running evaluation...")
    run = await orchestrator.run()

    print("\n4. Results summary:")
    print(f"   - Total tests: {len(run.results)}")
    print(f"   - Success rate: {run.success_rate:.2%}")

    # Get aggregated metrics (equivalent to old summary)
    metrics = orchestrator.get_aggregated_metrics()
    print(f"   - Mean Precision: {metrics.metrics.get('mean_precision', 0):.3f}")
    print(f"   - Mean Recall: {metrics.metrics.get('mean_recall', 0):.3f}")
    print(f"   - Mean F1: {metrics.metrics.get('mean_f1_score', 0):.3f}")

    return run


async def compare_old_vs_new():
    """
    Run same evaluation with old and new framework to compare results.
    """
    print("\nComparing Old vs New Framework")
    print("="*60)

    # Configuration for both
    test_params = {
        "test_file": "G1_instruction.json",
        "num_cases": 10,
        "add_noise_to_available": 50
    }

    # Run with new framework
    print("\n1. Running with NEW framework...")
    new_config = migrate_config(**test_params)
    new_orchestrator = EvaluationOrchestrator(new_config)
    new_run = await new_orchestrator.run()
    new_metrics = new_orchestrator.get_aggregated_metrics()

    # If you want to run with old framework (requires old evaluator to be available)
    try:
        from src.evaluation.toolbench_evaluator import ToolBenchEvaluator

        print("\n2. Running with OLD framework...")
        old_evaluator = ToolBenchEvaluator()
        old_results = await old_evaluator.run_evaluation(**test_params)

        # Compare results
        print("\n3. Comparison:")
        print(f"   Old F1: {old_results['summary']['metrics']['avg_f1_score']:.3f}")
        print(f"   New F1: {new_metrics.metrics.get('mean_f1_score', 0):.3f}")

    except ImportError:
        print("\n2. Old evaluator not available for comparison")
        print("   (This is expected if you've already migrated)")

    return new_run


async def compare_strategies_efficiently():
    """
    Example of using StrategyComparator for efficient multi-strategy comparison.
    This is a NEW CAPABILITY not available in the old evaluator!
    """
    print("\nEfficient Strategy Comparison using StrategyComparator")
    print("="*60)
    print("This feature indexes data ONCE and tests ALL strategies on the same data!")
    print("This is much more efficient than running separate evaluations.")

    # Base configuration (strategy will be overridden by comparator)
    base_config = EvaluationConfig(
        experiment_name="strategy_comparison",
        description="Comparing multiple search strategies efficiently",
        data=DataConfig(
            test_file="G2_instruction.json",
            num_cases=5,  # Small number for demo
            clear_collection_between_tests=True
        ),
        noise=NoiseConfig(
            add_noise_to_available=100,  # Add realistic noise
            noise_data_path=Path("toolbench_data/data/test_instruction")  # Default noise data path
        ),
        execution=ExecutionConfig(
            mode=TestExecutionMode.SEQUENTIAL,  # Each strategy runs sequentially
            timeout_per_test=30.0
        ),
        metrics=MetricsConfig(
            calculate_traditional_metrics=True,
            calculate_ranking_metrics=True,
            calculate_noise_metrics=True,
            optimize_threshold=False  # Don't optimize per strategy
        ),
        reporting=ReportingConfig(
            output_dir=Path("strategy_comparison_results"),
            generate_comparison_table=True,  # Generate comparison table
            export_formats=["json", "markdown", "html"]
        )
    )

    # Create the comparator
    print("\n1. Creating StrategyComparator...")
    comparator = StrategyComparator(base_config)

    # Define strategies to compare with strategy-specific configurations
    strategies_to_compare = [
        StrategyConfig(
            strategy=SearchStrategy.SEMANTIC,
            threshold=0.0,    # Low threshold for evaluation to include all tools
            max_tools=20     # High limit to accommodate available + noise tools
        ),
        StrategyConfig(
            strategy=SearchStrategy.HYBRID,
            threshold=0.0,    # Low threshold for evaluation
            max_tools=20
        ),
        StrategyConfig(
            strategy=SearchStrategy.HYBRID_CROSS_ENCODER,
            threshold=0.0,    # Low threshold for evaluation
            max_tools=10
        ),
        StrategyConfig(
            strategy=SearchStrategy.HYBRID_LTR,
            threshold=0.13,    # Low threshold for evaluation - LTR handles ranking
            max_tools=10
        )
    ]

    print(f"\n2. Strategies to compare: {[s.strategy.value for s in strategies_to_compare]}")
    print("   Configuration details:")
    for i, strategy_config in enumerate(strategies_to_compare, 1):
        print(f"     {i}. {strategy_config.strategy.value}: threshold={strategy_config.threshold}, max_tools={strategy_config.max_tools}")

    # Progress callback to show which strategy is being tested
    def progress_callback(strategy_name: str, completed: int, total: int):
        print(f"   {strategy_name}: {completed}/{total} tests completed ({completed/total*100:.1f}%)")

    print("\n3. Running comparison (indexes data once, tests all strategies)...")
    comparison_result = await comparator.compare_strategies(
        strategies_to_compare,
        progress_callback=progress_callback,
        parallel_strategies=False  # Set to True for parallel execution
    )

    print("\n4. Comparison Results:")
    print("   Best performers by metric:")

    # Find best strategy for each key metric
    for metric in ['mean_f1_score', 'mean_mrr', 'mean_ndcg@5']:
        best_strategy = None
        best_value = -1

        # comparison_result.comparison_metrics is run_id -> metrics dict
        for run_id, metrics in comparison_result.comparison_metrics.items():
            value = metrics.get(metric, 0)
            if value > best_value:
                best_value = value
                # Find the run with this ID to get strategy name
                for run in comparison_result.runs:
                    if run.id == run_id:
                        best_strategy = run.name.replace("strategy_", "")
                        break

        if best_strategy:
            print(f"   - {metric}: {best_strategy} ({best_value:.3f})")

    # Show summary comparison
    print("\n5. Summary Table:")
    print("   " + "-"*60)
    print(f"   {'Strategy':<30} {'F1 Score':<10} {'MRR':<10} {'NDCG@5':<10}")
    print("   " + "-"*60)

    # comparison_result.runs is a list of EvaluationRun objects
    for run in comparison_result.runs:
        strategy_name = run.name.replace("strategy_", "")
        metrics = comparison_result.comparison_metrics.get(run.id, {})

        f1 = metrics.get('mean_f1_score', 0)
        mrr = metrics.get('mean_mrr', 0)
        ndcg = metrics.get('mean_ndcg@5', 0)
        print(f"   {strategy_name:<30} {f1:<10.3f} {mrr:<10.3f} {ndcg:<10.3f}")

    print("   " + "-"*60)

    print("\n6. Key Benefits of StrategyComparator:")
    print("   ✓ Data indexed only ONCE (not 4 times for 4 strategies)")
    print("   ✓ All strategies tested on IDENTICAL data")
    print("   ✓ Fair comparison with same test conditions")
    print("   ✓ Option for parallel strategy execution")
    print("   ✓ Automatic comparison metrics and reports")

    return comparison_result


async def advanced_migration_example():
    """
    Example of migrating to use advanced features of the new framework.
    """
    print("\nAdvanced Migration: Using new framework features")
    print("="*60)

    # Start with basic migration
    base_config = migrate_config(
        test_file="G1_instruction.json",
        num_cases=20,
        add_noise_to_available=50
    )

    # Enhance with new features
    print("\n1. Adding parallel execution...")
    base_config.execution.mode = TestExecutionMode.PARALLEL
    base_config.execution.num_workers = 4

    print("2. Adding HTML reporting...")
    base_config.reporting.export_formats.append("html")

    print("3. Adding experiment tracking...")
    base_config.experiment_name = "advanced_migration_demo"
    base_config.tags = ["migration", "parallel", "noise-testing"]

    # Run with enhanced configuration
    orchestrator = EvaluationOrchestrator(
        base_config,
        use_cache=True,  # Enable caching
        track_experiments=True  # Enable experiment tracking
    )

    # Run with progress tracking
    def progress_callback(completed: int, total: int):
        print(f"   Progress: {completed}/{total} ({completed/total*100:.1f}%)")

    run = await orchestrator.run(progress_callback=progress_callback)

    print("\n4. New features utilized:")
    print(f"   - Parallel execution with {base_config.execution.num_workers} workers")
    print(f"   - HTML report generated")
    print(f"   - Results cached for faster re-runs")
    print(f"   - Experiment tracked for comparison")

    return run


def print_migration_guide():
    """
    Print a migration guide for users.
    """
    guide = """
    MIGRATION GUIDE: ToolBenchEvaluator → Evaluation Framework
    ==========================================================

    1. CONFIGURATION CHANGES
    ------------------------
    Old: Parameters passed to run_evaluation()
    New: Structured configuration using EvaluationConfig

    Benefits:
    - Type-safe configuration
    - Easy serialization/deserialization
    - Configuration reuse and sharing

    2. DATA LOADING
    ---------------
    Old: Built into evaluator
    New: Separate DataLoader classes

    Benefits:
    - Support for multiple data sources
    - Easy to add custom data loaders
    - Better separation of concerns

    3. EXECUTION MODES
    ------------------
    Old: Sequential only
    New: Sequential, Parallel, Batch

    Benefits:
    - Faster evaluation with parallel execution
    - Better resource utilization
    - Configurable retry and timeout

    4. METRICS
    ----------
    Old: Calculated inline
    New: Dedicated MetricsCalculator

    Benefits:
    - Centralized metric definitions
    - Easy to add new metrics
    - Statistical aggregation built-in

    5. REPORTING
    ------------
    Old: JSON output only
    New: JSON, CSV, Markdown, HTML

    Benefits:
    - Multiple export formats
    - Interactive HTML reports
    - Comparison tables

    6. NEW FEATURES
    ---------------
    - Result caching
    - Experiment tracking
    - Multi-run comparisons
    - Threshold optimization
    - Progress callbacks
    - StrategyComparator for efficient multi-strategy testing

    7. STRATEGY COMPARISON (NEW!)
    -----------------------------
    Old: Run separate evaluations for each strategy (inefficient)
    New: StrategyComparator - index once, test all strategies

    Benefits:
    - Single data indexing for all strategies
    - Identical test conditions for fair comparison
    - Parallel strategy execution option
    - Automatic comparison metrics

    MIGRATION STEPS
    ---------------
    1. Use migrate_config() to convert parameters
    2. Create EvaluationOrchestrator with config
    3. Call orchestrator.run() instead of evaluator.run_evaluation()
    4. Access results through the returned EvaluationRun object

    EXAMPLE
    -------
    # Old
    evaluator = ToolBenchEvaluator()
    results = await evaluator.run_evaluation(
        test_file="test.json",
        num_cases=50
    )

    # New
    config = migrate_config(
        test_file="test.json",
        num_cases=50
    )
    orchestrator = EvaluationOrchestrator(config)
    run = await orchestrator.run()
    """

    print(guide)


async def main():
    """
    Main function to run migration examples.
    """
    print("\n" + "="*80)
    print("EVALUATION FRAMEWORK MIGRATION EXAMPLES")
    print("="*80)

    # Print migration guide
    print_migration_guide()

    # Run migration examples
    print("\n" + "="*80)
    print("RUNNING MIGRATION EXAMPLES")
    print("="*80)

    # Basic migration
    print("\n[Example 1: Basic Migration]")
    run1 = await migrate_and_run()

    # Advanced migration
    # print("\n[Example 2: Advanced Migration]")
    # run2 = await advanced_migration_example()

    # Strategy comparison - NEW CAPABILITY!
    print("\n[Example 3: Efficient Strategy Comparison]")
    comparison = await compare_strategies_efficiently()

    # Old vs New comparison (optional - commented out by default)
    # print("\n[Example 4: Framework Comparison]")
    # run3 = await compare_old_vs_new()

    print("\n" + "="*80)
    print("MIGRATION COMPLETE!")
    print("="*80)
    print("\nThe new evaluation framework is ready to use.")
    print("See example_usage.py for more examples of new features.")
    print("\nKEY IMPROVEMENT: Use StrategyComparator for efficient multi-strategy testing!")

    # return [run1, run2, comparison]
    return [run1, comparison]


if __name__ == "__main__":
    asyncio.run(main())
