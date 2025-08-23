"""
Strategy comparison utilities for the evaluation framework.
Enables efficient comparison of multiple search strategies.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import uuid
from copy import deepcopy
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .config import EvaluationConfig, SearchStrategyConfig
from .models import (
    EvaluationRun,
    EvaluationResult,
    ComparisonResult,
    TestCase,
    TestSuite,
    TestStatus
)
from .data_loader import BaseDataLoader, DataLoaderFactory
from .test_runner import TestRunner, TestContext
from .metrics_calculator import MetricsCalculator
from .reporter import EvaluationReporter
from .orchestrator import ExperimentTracker

from src.services.search_service import SearchStrategy, SearchService
from src.services.search_pipeline_config import (
    SearchPipelineConfig, get_evaluation_config, get_two_stage_config, 
    get_production_config
)
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.core.models import ToolFilterRequest
from src.core.config import get_settings


@dataclass
class StrategyConfig:
    """
    Modern configuration for search strategies using SearchPipelineConfig.
    
    This is the primary evaluation configuration that leverages the comprehensive
    SearchPipelineConfig system for maximum flexibility and control.
    
    Examples:
        # Evaluation with standard config
        StrategyConfig(
            strategy=SearchStrategy.HYBRID_LTR,
            pipeline_config=get_evaluation_config(final_threshold=0.13),
            name="standard_ltr"
        )
        
        # Two-stage filtering evaluation  
        StrategyConfig(
            strategy=SearchStrategy.TWO_STAGE,
            pipeline_config=get_two_stage_config(
                stage1_threshold=0.10,
                stage2_threshold=0.15
            ),
            name="two_stage_aggressive"
        )
        
        # Custom pipeline for research
        StrategyConfig(
            strategy=SearchStrategy.HYBRID,
            pipeline_config=SearchPipelineConfig(
                enable_ltr=False,
                final_threshold=0.20,
                enable_confidence_cutoff=True
            ),
            name="research_hybrid"
        )

    Attributes:
        strategy: Search strategy identifier (for compatibility)
        pipeline_config: Comprehensive pipeline configuration (mandatory)
        name: Human-readable name for identification in results (mandatory)
    """
    strategy: SearchStrategy  # Strategy identifier (mainly for compatibility)
    pipeline_config: SearchPipelineConfig  # Comprehensive pipeline control
    name: str  # Human-readable name for results

    def get_display_name(self) -> str:
        """Get display name for results and logging."""
        return self.name or f"{self.strategy.value}_config"


class StrategyComparator:
    """
    Efficiently compares multiple search strategies on the same test data.

    Key features:
    - Indexes data once, tests all strategies
    - Ensures fair comparison (same test conditions)
    - Parallel strategy execution option
    - Detailed comparison metrics
    """

    def __init__(
        self,
        base_config: EvaluationConfig,
        track_experiments: bool = True
    ):
        """
        Initialize the strategy comparator.

        Args:
            base_config: Base configuration (strategy will be overridden)
            track_experiments: Whether to track experiments
        """
        self.base_config = base_config
        self.data_loader = DataLoaderFactory.create_loader(base_config.data)
        self.metrics_calculator = MetricsCalculator(base_config.metrics)
        self.reporter = EvaluationReporter(base_config.reporting)
        self.tracker = ExperimentTracker() if track_experiments else None

    async def compare_strategies(
        self,
        strategies: List[StrategyConfig],
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        parallel_strategies: bool = False
    ) -> ComparisonResult:
        """
        Compare multiple strategies on the same test data.

        Args:
            strategies: List of strategy configurations to compare
            progress_callback: Callback for progress updates (strategy_name, completed, total)
            parallel_strategies: Whether to run strategies in parallel

        Returns:
            Comparison results with detailed metrics
        """
        logger.info("="*60)
        logger.info(f"Strategy Comparison: {len(strategies)} strategies")
        logger.info("="*60)

        # Load test data once
        logger.info("Loading test data...")
        test_suite = self.data_loader.load_test_suite()
        logger.info(f"Loaded {test_suite.size} test cases")

        # Initialize shared services (but don't pre-index)
        logger.info("Initializing shared services...")
        test_context = await self._initialize_shared_context()

        # Run each strategy
        runs = []

        if parallel_strategies:
            # Run strategies in parallel
            logger.info("Running strategies in parallel...")
            strategy_tasks = [
                self._evaluate_strategy(
                    strategy_config, test_suite, test_context, progress_callback
                )
                for strategy_config in strategies
            ]
            runs = await asyncio.gather(*strategy_tasks)
        else:
            # Run strategies sequentially
            logger.info("Running strategies sequentially...")
            for strategy_config in strategies:
                run = await self._evaluate_strategy(
                    strategy_config, test_suite, test_context, progress_callback
                )
                runs.append(run)

        # Clean up shared context
        await test_context.cleanup()

        # Compare results
        logger.info("="*60)
        logger.info("Generating comparison...")
        comparison = self._create_comparison(runs)

        # Generate reports
        if self.base_config.reporting.generate_comparison_table:
            logger.info("Generating comparison reports...")
            self.reporter.export_comparison(comparison)

        # Track comparison
        if self.tracker:
            self.tracker.track_comparison(
                str(uuid.uuid4()),
                [run.id for run in runs],
                comparison.best_run_by_metric,
                {"strategies": [s.strategy.value for s in strategies]}
            )

        # Print summary
        self._print_comparison_summary(comparison)

        return comparison

    async def _initialize_shared_context(self) -> TestContext:
        """
        Initialize shared services for all strategies.

        Returns:
            Shared test context
        """
        settings = get_settings()

        # Initialize embedding service
        embedding_service = EmbeddingService(
            model=self.base_config.search.primary_embedding_model,
            api_key=settings.primary_embedding_api_key
        )

        # Initialize vector store
        dimension = settings.get_embedding_dimension(
            self.base_config.search.primary_embedding_model
        )
        vector_store = VectorStoreService(
            embedding_dimension=dimension,
            model_name=self.base_config.search.primary_embedding_model,
            similarity_threshold=self.base_config.search.primary_similarity_threshold
        )
        await vector_store.initialize()

        # Initialize search service
        search_service = SearchService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )

        # Load noise pool if configured
        noise_pool = None
        if self.base_config.noise.add_noise_to_available > 0:
            from .data_loader import NoiseDataLoader, ToolBenchDataLoader, DataConfig

            # Use noise config's data path instead of settings
            data_config = DataConfig(
                data_source=self.base_config.noise.noise_data_source,
                data_path=self.base_config.noise.noise_data_path
            )
            base_loader = ToolBenchDataLoader(data_config)
            noise_loader = NoiseDataLoader(data_config, base_loader)
            noise_pool = noise_loader.load_noise_pool(
                target_size=self.base_config.noise.noise_pool_size
            )

        return TestContext(
            vector_store=vector_store,
            embedding_service=embedding_service,
            search_service=search_service,
            search_config=self.base_config.search,
            noise_config=self.base_config.noise,
            noise_pool=noise_pool
        )


    async def _evaluate_strategy(
        self,
        strategy_config: StrategyConfig,
        test_suite: TestSuite,
        test_context: TestContext,
        progress_callback: Optional[Callable] = None
    ) -> EvaluationRun:
        """
        Evaluate a single strategy on the test suite.

        Args:
            strategy_config: Strategy configuration to evaluate
            test_suite: Test suite
            test_context: Shared test context
            progress_callback: Progress callback

        Returns:
            Evaluation run results
        """
        strategy_name = strategy_config.get_display_name()
        logger.info(f"Evaluating strategy: {strategy_name}")

        # Use the shared test context directly (no strategy-specific config needed)
        # The pipeline configuration will be passed to search_with_config
        strategy_test_context = test_context

        run_id = str(uuid.uuid4())
        run = EvaluationRun(
            id=run_id,
            name=f"strategy_{strategy_name}",
            test_suite=test_suite,
            results=[],
            config={
                **self.base_config.to_dict(),
                "pipeline_config": {
                    "strategy": strategy_config.strategy.value,
                    "name": strategy_config.name,
                    "pipeline_params": strategy_config.pipeline_config.__dict__
                }
            },
            start_time=datetime.now()
        )

        results = []

        for i, test_case in enumerate(test_suite.test_cases):
            # Run test with this strategy
            result = await self._run_single_test(
                test_case,
                strategy_config,
                strategy_test_context
            )
            results.append(result)

            # Progress callback
            if progress_callback:
                progress_callback(strategy_name, i + 1, test_suite.size)

        # Calculate comprehensive metrics
        noise_tool_names = None
        if strategy_test_context.noise_pool:
            noise_tool_names = {tool["name"] for tool in strategy_test_context.noise_pool}

        enhanced_results = []
        for result in results:
            full_metrics = self.metrics_calculator.calculate_metrics_for_result(
                result,
                noise_tool_names
            )

            enhanced_result = EvaluationResult(
                test_case=result.test_case,
                recommended_tools=result.recommended_tools,
                metrics=full_metrics,
                execution_time_ms=result.execution_time_ms,
                status=result.status,
                error=result.error,
                metadata={
                    **result.metadata, 
                    "strategy": strategy_config.strategy.value,
                    "strategy_name": strategy_name,
                    "pipeline_config": strategy_config.pipeline_config.__dict__
                }
            )
            enhanced_results.append(enhanced_result)

        run.results = enhanced_results
        run.end_time = datetime.now()

        # Print summary for this strategy
        aggregated = self.metrics_calculator.aggregate_metrics(
            enhanced_results,
            noise_tool_names
        )

        logger.info(f"  {strategy_name} Results:")
        logger.info(f"    Mean F1: {aggregated.metrics.get('mean_f1_score', 0):.3f}")
        logger.info(f"    Mean MRR: {aggregated.metrics.get('mean_mrr', 0):.3f}")
        logger.info(f"    Mean P@1: {aggregated.metrics.get('mean_p@1', 0):.3f}")

        return run

    async def _run_single_test(
        self,
        test_case: TestCase,
        strategy_config: StrategyConfig,
        test_context: TestContext
    ) -> EvaluationResult:
        """
        Run a single test with a specific strategy configuration.

        Args:
            test_case: Test case to run
            strategy_config: Strategy configuration to use
            test_context: Test context (already configured for this strategy)

        Returns:
            Evaluation result
        """
        import time
        from src.core.models import Tool
        import random

        start_time = time.time()

        try:
            # Clear collection if configured (to ensure clean indexing per test)
            if self.base_config.data.clear_collection_between_tests:
                try:
                    # Delete and recreate collection for clean state
                    test_context.vector_store.client.delete_collection(test_context.vector_store.collection_name)
                    await test_context.vector_store.initialize()
                    # print(f"   [DEBUG] Cleared collection for test: {test_case.id}")
                except Exception as e:
                    # Log the error but continue - collection might not exist
                    logger.warning(f"Failed to clear collection: {e}")
                    # Try to initialize anyway
                    await test_context.vector_store.initialize()

            # Index tools for this test case (same as TestRunner)
            logger.debug(f"Test case has {len(test_case.available_tools)} original tools")
            # Show the names of the original tools
            if test_case.available_tools:
                original_tool_names = [tool.get("name", "NO_NAME") for tool in test_case.available_tools[:3]]
                logger.debug(f"Original tool names: {original_tool_names}")
            indexed_tools = await self._index_tools(test_case.available_tools, test_context)
            logger.debug(f"Indexed {len(indexed_tools)} tools from test case")

            # Add noise tools if configured
            tools_with_noise = indexed_tools.copy()
            if test_context.noise_config.add_noise_to_available > 0 and test_context.noise_pool:
                # Get tool names to avoid duplicates
                test_tool_names = {tool["name"] for tool in indexed_tools}

                # Filter noise pool
                available_noise = [
                    tool for tool in test_context.noise_pool
                    if tool["name"] not in test_tool_names
                ]

                # Sample noise tools
                num_noise = min(
                    test_context.noise_config.add_noise_to_available,
                    len(available_noise)
                )

                if num_noise > 0:
                    noise_tools = random.sample(available_noise, num_noise)
                    logger.debug(f"Adding {len(noise_tools)} noise tools")

                    # INDEX the noise tools in the vector store!
                    await self._index_tools(noise_tools, test_context)

                    tools_with_noise.extend(noise_tools)
                    logger.debug(f"Total available_tools: {len(tools_with_noise)} (should be {len(indexed_tools) + len(noise_tools)})")

            # Convert to Tool objects
            tools = [Tool(**tool_dict) for tool_dict in tools_with_noise]

            # Create request - use pipeline config's final_limit for max_tools
            request = ToolFilterRequest(
                messages=[{"role": "user", "content": test_case.query}],
                available_tools=tools,
                max_tools=strategy_config.pipeline_config.final_limit
            )

            # Perform search with pipeline configuration
            recommended_tools = await test_context.search_service.search_with_config(
                messages=request.messages,
                available_tools=request.available_tools,
                config=strategy_config.pipeline_config
            )

            # Debug: Log the structure of recommended_tools
            logger.debug(f"Search returned {len(recommended_tools)} tools")
            if recommended_tools and len(recommended_tools) > 0:
                logger.debug(f"First tool structure: {recommended_tools[0].keys() if isinstance(recommended_tools[0], dict) else type(recommended_tools[0])}")
                if isinstance(recommended_tools[0], dict):
                    # Check for different possible name fields
                    first_tool = recommended_tools[0]
                    logger.debug(f"First tool has 'name': {first_tool.get('name', 'NO_NAME')}")
                    logger.debug(f"First tool has 'tool_name': {first_tool.get('tool_name', 'NO_TOOL_NAME')}")
                    # Show first 3 tool names
                    for i, tool in enumerate(recommended_tools[:3]):
                        tool_name = tool.get("tool_name") or tool.get("name") or "UNKNOWN"
                        logger.debug(f"Tool {i}: {tool_name} (score: {tool.get('score', 'NO_SCORE')})")

            execution_time = (time.time() - start_time) * 1000

            # Calculate basic metrics
            expected_set = set(test_case.expected_tools)
            logger.debug(f"Expected tools ({len(expected_set)} total): {list(expected_set)}")  # Show all expected
            
            # Try both 'name' and 'tool_name' fields
            recommended_set = set()
            for tool in recommended_tools:
                if isinstance(tool, dict):
                    # Try 'tool_name' first, then 'name'
                    tool_name = tool.get("tool_name") or tool.get("name")
                    if tool_name:
                        recommended_set.add(tool_name)
            
            logger.debug(f"Recommended tools ({len(recommended_set)} total): {list(recommended_set)[:5]}...")  # Show first 5 recommended
            
            # Check for any intersection
            intersection = expected_set & recommended_set
            if intersection:
                logger.debug(f"✓ Found matches: {intersection}")
            else:
                logger.debug("✗ NO MATCHES FOUND!")
                # Check if it's a naming issue
                for exp in list(expected_set)[:2]:
                    logger.debug(f"  Expected: '{exp}'")
                    for rec in list(recommended_set)[:2]:
                        if exp.lower() in rec.lower() or rec.lower() in exp.lower():
                            logger.debug(f"    Possible match: '{rec}'")
                            break

            true_positives = len(expected_set & recommended_set)
            false_positives = len(recommended_set - expected_set)
            false_negatives = len(expected_set - recommended_set)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            logger.debug(f"Basic metrics: TP={true_positives}, FP={false_positives}, FN={false_negatives}")
            logger.debug(f"Basic metrics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}")

            return EvaluationResult(
                test_case=test_case,
                recommended_tools=recommended_tools,
                metrics={
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                },
                execution_time_ms=execution_time,
                status=TestStatus.COMPLETED
            )

        except Exception as e:
            return EvaluationResult(
                test_case=test_case,
                recommended_tools=[],
                metrics={},
                execution_time_ms=(time.time() - start_time) * 1000,
                status=TestStatus.FAILED,
                error=str(e)
            )

    async def _index_tools(
        self,
        tools: List[Dict[str, Any]],
        test_context: TestContext
    ) -> List[Dict[str, Any]]:
        """
        Index tools in the vector store (same as TestRunner).

        Args:
            tools: List of tool dictionaries to index
            test_context: Test context with services

        Returns:
            List of indexed tools
        """
        if not tools:
            logger.debug("No tools to index")
            return []

        logger.debug(f"Indexing {len(tools)} tools...")

        from src.services.embedding_enhancer import ToolEmbeddingEnhancer
        from src.core.models import Tool

        enhancer = ToolEmbeddingEnhancer()

        # Generate embeddings for tools
        tool_texts = []
        valid_tools = []
        for i, tool_dict in enumerate(tools):
            try:
                tool = Tool(**tool_dict)
                text = enhancer.tool_to_rich_text(tool)
                tool_texts.append(text)
                valid_tools.append(tool_dict)
                if i < 3:  # Show first 3 tool names for debugging
                    logger.debug(f"Tool {i}: {tool_dict.get('name', 'NO_NAME')}")
            except Exception as e:
                logger.debug(f"Failed to process tool {i}: {e}")
                continue

        if not tool_texts:
            logger.debug("No valid tools to index after processing")
            return []

        logger.debug(f"Generated {len(tool_texts)} embeddings for {len(valid_tools)} tools")

        # Get embeddings
        try:
            embeddings = await test_context.embedding_service.embed_batch(tool_texts)
            logger.debug(f"Embeddings generated: {len(embeddings)} vectors")
        except Exception as e:
            logger.debug(f"Failed to generate embeddings: {e}")
            return []

        # Index in vector store
        try:
            await test_context.vector_store.index_tools_batch(valid_tools, embeddings)
            logger.debug(f"Successfully indexed {len(valid_tools)} tools in vector store")
            
            # Return the successfully indexed tools
            return valid_tools

        except Exception as e:
            logger.error(f"Failed to index tools in vector store: {e}")
            return []

    def _create_comparison(self, runs: List[EvaluationRun]) -> ComparisonResult:
        """
        Create comparison result from multiple runs.

        Args:
            runs: List of evaluation runs

        Returns:
            Comparison result
        """
        # Calculate comparison metrics
        comparison_metrics = self.metrics_calculator.compare_runs(runs)

        # Determine best performers
        best_run_by_metric = {}

        for metric_name in comparison_metrics.get(runs[0].id, {}).keys():
            best_value = None
            best_run_id = None

            for run_id, metrics in comparison_metrics.items():
                value = metrics.get(metric_name)

                if value is not None:
                    is_better = (
                        best_value is None or
                        (value > best_value if "loss" not in metric_name.lower() else value < best_value)
                    )

                    if is_better:
                        best_value = value
                        best_run_id = run_id

            if best_run_id:
                best_run_by_metric[metric_name] = best_run_id

        # Statistical tests (optional, placeholder for now)
        statistical_tests = {}

        return ComparisonResult(
            runs=runs,
            comparison_metrics=comparison_metrics,
            best_run_by_metric=best_run_by_metric,
            statistical_tests=statistical_tests,
            metadata={
                "comparison_time": datetime.now().isoformat(),
                "num_strategies": len(runs)
            }
        )

    def _print_comparison_summary(self, comparison: ComparisonResult):
        """
        Print a summary of the comparison results.

        Args:
            comparison: Comparison results
        """
        logger.info("="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)

        # Create summary table
        logger.info("| Strategy | F1 Score | MRR | P@1 | NDCG@10 |")
        logger.info("|----------|----------|-----|-----|---------|")

        for run in comparison.runs:
            strategy = run.name.replace("strategy_", "")
            metrics = comparison.comparison_metrics.get(run.id, {})

            f1 = metrics.get("mean_f1_score", 0)
            mrr = metrics.get("mean_mrr", 0)
            p1 = metrics.get("mean_p@1", 0)
            ndcg = metrics.get("mean_ndcg@10", 0)

            # Mark best performers
            f1_mark = " ⭐" if comparison.best_run_by_metric.get("mean_f1_score") == run.id else ""
            mrr_mark = " ⭐" if comparison.best_run_by_metric.get("mean_mrr") == run.id else ""
            p1_mark = " ⭐" if comparison.best_run_by_metric.get("mean_p@1") == run.id else ""
            ndcg_mark = " ⭐" if comparison.best_run_by_metric.get("mean_ndcg@10") == run.id else ""

            logger.info(f"| {strategy:8s} | {f1:.3f}{f1_mark:3s} | {mrr:.3f}{mrr_mark:3s} | "
                  f"{p1:.3f}{p1_mark:3s} | {ndcg:.3f}{ndcg_mark:3s} |")

        logger.info("⭐ = Best performer for that metric")

        # Overall winner
        from collections import Counter
        winner_counts = Counter(comparison.best_run_by_metric.values())
        if winner_counts:
            overall_winner_id = winner_counts.most_common(1)[0][0]
            overall_winner = next((r.name for r in comparison.runs if r.id == overall_winner_id), "Unknown")
            logger.info(f"Overall Winner: {overall_winner.replace('strategy_', '')} "
                  f"(best in {winner_counts[overall_winner_id]} metrics)")


class MultiConfigRunner:
    """
    Runs multiple configurations efficiently.
    Useful for hyperparameter tuning or configuration comparison.
    """

    def __init__(self, base_config: EvaluationConfig):
        """
        Initialize multi-config runner.

        Args:
            base_config: Base configuration template
        """
        self.base_config = base_config

    async def run_configs(
        self,
        config_variations: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[EvaluationRun]:
        """
        Run multiple configuration variations.

        Args:
            config_variations: List of config modifications
            parallel: Whether to run in parallel

        Returns:
            List of evaluation runs
        """
        from .orchestrator import EvaluationOrchestrator

        runs = []

        for i, variation in enumerate(config_variations, 1):
            # Create modified config
            config = deepcopy(self.base_config)

            # Apply variations
            for key, value in variation.items():
                if "." in key:
                    # Handle nested keys like "search.primary_similarity_threshold"
                    parts = key.split(".")
                    obj = config
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, parts[-1], value)
                else:
                    setattr(config, key, value)

            # Update experiment name
            config.experiment_name = f"{config.experiment_name}_variation_{i}"

            # Run evaluation
            orchestrator = EvaluationOrchestrator(config)
            run = await orchestrator.run()
            runs.append(run)

        return runs
