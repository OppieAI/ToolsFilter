"""
Main orchestrator for the evaluation framework.
Coordinates all components and provides high-level API.
"""

import asyncio
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import hashlib

from .config import EvaluationConfig
from .data_loader import BaseDataLoader, DataLoaderFactory, NoiseDataLoader
from .models import (
    EvaluationRun,
    EvaluationResult,
    ComparisonResult,
    TestCase,
    TestSuite
)
from .test_runner import TestRunner
from .metrics_calculator import MetricsCalculator, AggregatedMetrics
from .reporter import EvaluationReporter, ReportFormat


class ExperimentTracker:
    """
    Tracks experiment history and metadata.
    """
    
    def __init__(self, tracking_dir: Path = Path("experiment_tracking")):
        """
        Initialize experiment tracker.
        
        Args:
            tracking_dir: Directory for tracking files
        """
        self.tracking_dir = tracking_dir
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.tracking_dir / "experiment_history.json"
        self._load_history()
    
    def _load_history(self):
        """Load experiment history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                "experiments": [],
                "comparisons": []
            }
    
    def _save_history(self):
        """Save experiment history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def track_experiment(
        self,
        run_id: str,
        config: EvaluationConfig,
        metrics: Dict[str, float],
        metadata: Dict[str, Any]
    ):
        """
        Track an experiment run.
        
        Args:
            run_id: Unique run identifier
            config: Configuration used
            metrics: Key metrics from the run
            metadata: Additional metadata
        """
        experiment = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config_hash": self._hash_config(config),
            "config": config.to_dict(),
            "metrics": metrics,
            "metadata": metadata
        }
        
        self.history["experiments"].append(experiment)
        self._save_history()
    
    def track_comparison(
        self,
        comparison_id: str,
        run_ids: List[str],
        best_performers: Dict[str, str],
        metadata: Dict[str, Any]
    ):
        """
        Track a comparison between runs.
        
        Args:
            comparison_id: Unique comparison identifier
            run_ids: IDs of runs being compared
            best_performers: Best run for each metric
            metadata: Additional metadata
        """
        comparison = {
            "comparison_id": comparison_id,
            "timestamp": datetime.now().isoformat(),
            "run_ids": run_ids,
            "best_performers": best_performers,
            "metadata": metadata
        }
        
        self.history["comparisons"].append(comparison)
        self._save_history()
    
    def _hash_config(self, config: EvaluationConfig) -> str:
        """Generate hash of configuration for deduplication."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def find_similar_experiments(
        self,
        config: EvaluationConfig,
        metric: str = "mean_f1_score"
    ) -> List[Dict[str, Any]]:
        """
        Find experiments with similar configuration.
        
        Args:
            config: Configuration to match
            metric: Metric to sort by
            
        Returns:
            List of similar experiments
        """
        config_hash = self._hash_config(config)
        similar = [
            exp for exp in self.history["experiments"]
            if exp["config_hash"] == config_hash
        ]
        
        # Sort by metric
        similar.sort(key=lambda x: x["metrics"].get(metric, 0), reverse=True)
        
        return similar


class ResultCache:
    """
    Caches evaluation results to avoid re-computation.
    """
    
    def __init__(self, cache_dir: Path = Path(".evaluation_cache")):
        """
        Initialize result cache.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_cache_key(
        self,
        config: EvaluationConfig,
        test_suite: TestSuite
    ) -> str:
        """
        Generate cache key for configuration and test suite.
        
        Args:
            config: Evaluation configuration
            test_suite: Test suite
            
        Returns:
            Cache key
        """
        # Combine config and test suite info for unique key
        key_data = {
            "config": config.to_dict(),
            "test_suite": {
                "name": test_suite.name,
                "size": test_suite.size,
                "test_ids": [tc.id for tc in test_suite.test_cases[:10]]  # Sample
            }
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[EvaluationRun]:
        """
        Get cached results.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached evaluation run or None
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return EvaluationRun.from_dict(data)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        return None
    
    def put(self, cache_key: str, run: EvaluationRun):
        """
        Cache evaluation results.
        
        Args:
            cache_key: Cache key
            run: Evaluation run to cache
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(run.to_dict(), f, default=str)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class EvaluationOrchestrator:
    """
    Main orchestrator that coordinates the entire evaluation process.
    """
    
    def __init__(
        self,
        config: EvaluationConfig,
        data_loader: Optional[BaseDataLoader] = None,
        use_cache: bool = True,
        track_experiments: bool = True
    ):
        """
        Initialize the evaluation orchestrator.
        
        Args:
            config: Evaluation configuration
            data_loader: Optional custom data loader
            use_cache: Whether to use result caching
            track_experiments: Whether to track experiments
        """
        self.config = config
        
        # Initialize components
        self.data_loader = data_loader or DataLoaderFactory.create_loader(config.data)
        self.test_runner = TestRunner(
            config.execution,
            config.search,
            config.noise
        )
        self.metrics_calculator = MetricsCalculator(config.metrics)
        self.reporter = EvaluationReporter(config.reporting)
        
        # Optional components
        self.cache = ResultCache() if use_cache else None
        self.tracker = ExperimentTracker() if track_experiments else None
        
        # State
        self.current_run: Optional[EvaluationRun] = None
        self.runs_history: List[EvaluationRun] = []
    
    async def run(
        self,
        force_rerun: bool = False,
        progress_callback: Optional[callable] = None
    ) -> EvaluationRun:
        """
        Run the complete evaluation pipeline.
        
        Args:
            force_rerun: Force re-evaluation even if cached
            progress_callback: Optional callback for progress updates
            
        Returns:
            Completed evaluation run
        """
        print(f"\n{'='*60}")
        print(f"Starting Evaluation: {self.config.experiment_name}")
        print(f"{'='*60}\n")
        
        # Load test suite
        print("Loading test data...")
        test_suite = self.data_loader.load_test_suite()
        print(f"Loaded {test_suite.size} test cases")
        
        # Check cache
        cache_key = None
        if self.cache and not force_rerun:
            cache_key = self.cache.get_cache_key(self.config, test_suite)
            cached_run = self.cache.get(cache_key)
            
            if cached_run:
                print("Found cached results, using cache...")
                self.current_run = cached_run
                self.runs_history.append(cached_run)
                return cached_run
        
        # Create run
        run_id = str(uuid.uuid4())
        run = EvaluationRun(
            id=run_id,
            name=self.config.experiment_name,
            test_suite=test_suite,
            results=[],
            config=self.config.to_dict(),
            start_time=datetime.now()
        )
        
        # Execute tests
        print(f"\nExecuting tests using {self.config.execution.mode.value} mode...")
        
        try:
            # Run tests
            results = await self.test_runner.run(
                test_suite.test_cases,
                progress_callback=progress_callback
            )
            
            # Calculate additional metrics if needed
            print("\nCalculating metrics...")
            
            # Get noise tool names if applicable
            noise_tool_names = None
            if self.config.noise.add_noise_to_available > 0:
                # Get noise tools from runner context
                if self.test_runner.test_context and self.test_runner.test_context.noise_pool:
                    noise_tool_names = {
                        tool["name"] for tool in self.test_runner.test_context.noise_pool
                    }
            
            # Enhance results with full metrics
            enhanced_results = []
            for result in results:
                # Calculate comprehensive metrics
                full_metrics = self.metrics_calculator.calculate_metrics_for_result(
                    result,
                    noise_tool_names
                )
                
                # Update result with full metrics
                enhanced_result = EvaluationResult(
                    test_case=result.test_case,
                    recommended_tools=result.recommended_tools,
                    metrics=full_metrics,
                    execution_time_ms=result.execution_time_ms,
                    status=result.status,
                    error=result.error,
                    metadata=result.metadata
                )
                enhanced_results.append(enhanced_result)
            
            run.results = enhanced_results
            run.end_time = datetime.now()
            
            # Cache results
            if self.cache and cache_key:
                self.cache.put(cache_key, run)
            
            # Track experiment
            if self.tracker:
                aggregated = self.metrics_calculator.aggregate_metrics(
                    enhanced_results,
                    noise_tool_names
                )
                
                self.tracker.track_experiment(
                    run_id,
                    self.config,
                    aggregated.metrics,
                    {
                        "test_suite": test_suite.name,
                        "num_tests": test_suite.size,
                        "duration_seconds": run.duration_seconds
                    }
                )
            
            # Store run
            self.current_run = run
            self.runs_history.append(run)
            
            # Generate summary
            print("\n" + self.reporter.generate_summary(run))
            
            # Export reports
            if self.config.reporting.save_detailed_results:
                print("\nGenerating reports...")
                exported = await self.export_results(run)
                print(f"Reports saved to: {self.config.reporting.output_dir}")
            
            return run
            
        finally:
            # Cleanup
            await self.test_runner.cleanup()
    
    async def export_results(
        self,
        run: Optional[EvaluationRun] = None,
        formats: Optional[List[ReportFormat]] = None
    ) -> Dict[ReportFormat, Path]:
        """
        Export evaluation results.
        
        Args:
            run: Evaluation run to export (None = current run)
            formats: Export formats (None = use config)
            
        Returns:
            Dictionary mapping format to output path
        """
        if run is None:
            run = self.current_run
        
        if run is None:
            raise ValueError("No evaluation run to export")
        
        return self.reporter.export_run(run, formats)
    
    async def compare_runs(
        self,
        runs: Optional[List[EvaluationRun]] = None,
        metric: str = "f1_score"
    ) -> ComparisonResult:
        """
        Compare multiple evaluation runs.
        
        Args:
            runs: List of runs to compare (None = all history)
            metric: Primary metric for comparison
            
        Returns:
            Comparison results
        """
        if runs is None:
            runs = self.runs_history
        
        if len(runs) < 2:
            raise ValueError("Need at least 2 runs to compare")
        
        print(f"\nComparing {len(runs)} evaluation runs...")
        
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
                    # Higher is better for most metrics
                    is_better = (
                        best_value is None or
                        (value > best_value if "loss" not in metric_name.lower() else value < best_value)
                    )
                    
                    if is_better:
                        best_value = value
                        best_run_id = run_id
            
            if best_run_id:
                best_run_by_metric[metric_name] = best_run_id
        
        # Create comparison result
        comparison = ComparisonResult(
            runs=runs,
            comparison_metrics=comparison_metrics,
            best_run_by_metric=best_run_by_metric,
            metadata={
                "comparison_time": datetime.now().isoformat(),
                "primary_metric": metric
            }
        )
        
        # Track comparison
        if self.tracker:
            comparison_id = str(uuid.uuid4())
            self.tracker.track_comparison(
                comparison_id,
                [run.id for run in runs],
                best_run_by_metric,
                {"primary_metric": metric}
            )
        
        # Export comparison
        if self.config.reporting.generate_comparison_table:
            print("\nGenerating comparison reports...")
            exported = self.reporter.export_comparison(comparison)
            print(f"Comparison reports saved to: {self.config.reporting.output_dir}")
        
        # Print summary
        print("\n" + self.reporter.generate_comparison_summary(comparison))
        
        return comparison
    
    async def run_multiple_configs(
        self,
        configs: List[EvaluationConfig]
    ) -> List[EvaluationRun]:
        """
        Run evaluations with multiple configurations.
        
        Args:
            configs: List of configurations to evaluate
            
        Returns:
            List of evaluation runs
        """
        runs = []
        
        for i, config in enumerate(configs, 1):
            print(f"\n{'='*60}")
            print(f"Running configuration {i}/{len(configs)}: {config.experiment_name}")
            print(f"{'='*60}")
            
            # Create new orchestrator for each config
            orchestrator = EvaluationOrchestrator(
                config,
                use_cache=self.cache is not None,
                track_experiments=self.tracker is not None
            )
            
            run = await orchestrator.run()
            runs.append(run)
            self.runs_history.append(run)
        
        # Compare all runs
        if len(runs) > 1:
            await self.compare_runs(runs)
        
        return runs
    
    def get_aggregated_metrics(
        self,
        run: Optional[EvaluationRun] = None
    ) -> AggregatedMetrics:
        """
        Get aggregated metrics for a run.
        
        Args:
            run: Evaluation run (None = current run)
            
        Returns:
            Aggregated metrics
        """
        if run is None:
            run = self.current_run
        
        if run is None:
            raise ValueError("No evaluation run available")
        
        # Get noise tools if applicable
        noise_tool_names = None
        if self.config.noise.add_noise_to_available > 0:
            if self.test_runner.test_context and self.test_runner.test_context.noise_pool:
                noise_tool_names = {
                    tool["name"] for tool in self.test_runner.test_context.noise_pool
                }
        
        return self.metrics_calculator.aggregate_metrics(
            run.results,
            noise_tool_names
        )
    
    def find_optimal_threshold(
        self,
        run: Optional[EvaluationRun] = None,
        metric: str = "f1_score"
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal threshold for a run.
        
        Args:
            run: Evaluation run (None = current run)
            metric: Metric to optimize
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        if run is None:
            run = self.current_run
        
        if run is None:
            raise ValueError("No evaluation run available")
        
        # Extract all scores and expected tools
        all_scores = []
        expected_tools = []
        
        for result in run.results:
            # Get all scores from metadata if available
            if "all_scores" in result.metadata:
                all_scores.append(result.metadata["all_scores"])
                expected_tools.append(set(result.test_case.expected_tools))
        
        if not all_scores:
            print("Warning: No score data available for threshold optimization")
            return self.config.search.primary_similarity_threshold, {}
        
        return self.metrics_calculator.find_optimal_threshold(
            all_scores,
            expected_tools,
            metric
        )