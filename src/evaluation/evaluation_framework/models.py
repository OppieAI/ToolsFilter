"""
Data models for the evaluation framework.
Provides immutable, type-safe representations of test cases and results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from enum import Enum
import hashlib
import json


class TestStatus(Enum):
    """Status of a test execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class TestCase:
    """
    Immutable representation of a single test case.
    
    Attributes:
        id: Unique identifier for the test case
        query: The user query/prompt
        expected_tools: List of expected tool names
        available_tools: List of all available tools for this test
        metadata: Additional metadata (dataset source, category, etc.)
    """
    id: str
    query: str
    expected_tools: List[str]
    available_tools: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def expected_set(self) -> Set[str]:
        """Get expected tools as a set for efficient comparison."""
        return set(self.expected_tools)
    
    @property
    def num_available_tools(self) -> int:
        """Get count of available tools."""
        return len(self.available_tools)
    
    @property
    def num_expected_tools(self) -> int:
        """Get count of expected tools."""
        return len(self.expected_tools)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "query": self.query,
            "expected_tools": self.expected_tools,
            "available_tools": self.available_tools,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestCase':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            query=data["query"],
            expected_tools=data["expected_tools"],
            available_tools=data["available_tools"],
            metadata=data.get("metadata", {})
        )
    
    def __hash__(self) -> int:
        """Hash based on ID for use in sets/dicts."""
        return hash(self.id)


@dataclass(frozen=True)
class MetricValue:
    """
    Immutable representation of a single metric value.
    
    Attributes:
        name: Metric name
        value: Metric value
        metadata: Additional information about the metric
    """
    name: str
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata
        }


@dataclass(frozen=True)
class EvaluationResult:
    """
    Immutable representation of a single test evaluation result.
    
    Attributes:
        test_case: The test case that was evaluated
        recommended_tools: Tools recommended by the system
        metrics: Calculated metrics
        execution_time_ms: Time taken to execute the test
        status: Test execution status
        error: Error message if test failed
        metadata: Additional result metadata
    """
    test_case: TestCase
    recommended_tools: List[Dict[str, Any]]
    metrics: Dict[str, float]
    execution_time_ms: float
    status: TestStatus = TestStatus.COMPLETED
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def recommended_set(self) -> Set[str]:
        """Get recommended tools as a set."""
        return {
            tool.get("name", "")
            for tool in self.recommended_tools
            if tool.get("name")
        }
    
    @property
    def is_successful(self) -> bool:
        """Check if the test was successful."""
        return self.status == TestStatus.COMPLETED and self.error is None
    
    @property
    def precision(self) -> float:
        """Get precision metric."""
        return self.metrics.get("precision", 0.0)
    
    @property
    def recall(self) -> float:
        """Get recall metric."""
        return self.metrics.get("recall", 0.0)
    
    @property
    def f1_score(self) -> float:
        """Get F1 score."""
        return self.metrics.get("f1_score", 0.0)
    
    @property
    def mrr(self) -> float:
        """Get Mean Reciprocal Rank."""
        return self.metrics.get("mrr", 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_case": self.test_case.to_dict(),
            "recommended_tools": self.recommended_tools,
            "metrics": self.metrics,
            "execution_time_ms": self.execution_time_ms,
            "status": self.status.value,
            "error": self.error,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create from dictionary."""
        return cls(
            test_case=TestCase.from_dict(data["test_case"]),
            recommended_tools=data["recommended_tools"],
            metrics=data["metrics"],
            execution_time_ms=data["execution_time_ms"],
            status=TestStatus(data.get("status", "completed")),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


@dataclass
class TestSuite:
    """
    Collection of test cases with metadata.
    
    Attributes:
        name: Name of the test suite
        test_cases: List of test cases
        metadata: Suite metadata (source, version, etc.)
    """
    name: str
    test_cases: List[TestCase]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """Get number of test cases."""
        return len(self.test_cases)
    
    def get_test_case(self, test_id: str) -> Optional[TestCase]:
        """Get test case by ID."""
        for test_case in self.test_cases:
            if test_case.id == test_id:
                return test_case
        return None
    
    def filter_by_metadata(self, key: str, value: Any) -> List[TestCase]:
        """Filter test cases by metadata."""
        return [
            tc for tc in self.test_cases
            if tc.metadata.get(key) == value
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestSuite':
        """Create from dictionary."""
        return cls(
            name=data["name"],
            test_cases=[TestCase.from_dict(tc) for tc in data["test_cases"]],
            metadata=data.get("metadata", {})
        )


@dataclass
class EvaluationRun:
    """
    Complete evaluation run with all results and metadata.
    
    Attributes:
        id: Unique run identifier
        name: Run name
        test_suite: The test suite that was evaluated
        results: List of evaluation results
        config: Configuration used for the run
        start_time: When the run started
        end_time: When the run completed
        metadata: Additional run metadata
    """
    id: str
    name: str
    test_suite: TestSuite
    results: List[EvaluationResult]
    config: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Get run duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def num_successful(self) -> int:
        """Count successful tests."""
        return sum(1 for r in self.results if r.is_successful)
    
    @property
    def num_failed(self) -> int:
        """Count failed tests."""
        return sum(1 for r in self.results if not r.is_successful)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if not self.results:
            return 0.0
        return self.num_successful / len(self.results)
    
    def get_result_by_test_id(self, test_id: str) -> Optional[EvaluationResult]:
        """Get result for a specific test case."""
        for result in self.results:
            if result.test_case.id == test_id:
                return result
        return None
    
    def aggregate_metrics(self) -> Dict[str, float]:
        """
        Aggregate metrics across all successful results.
        Returns mean values for each metric.
        """
        if not self.results:
            return {}
        
        successful_results = [r for r in self.results if r.is_successful]
        if not successful_results:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for result in successful_results:
            all_metrics.update(result.metrics.keys())
        
        # Calculate mean for each metric
        aggregated = {}
        for metric_name in all_metrics:
            values = [
                r.metrics.get(metric_name, 0.0)
                for r in successful_results
            ]
            aggregated[f"mean_{metric_name}"] = sum(values) / len(values)
        
        return aggregated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "test_suite": self.test_suite.to_dict(),
            "results": [r.to_dict() for r in self.results],
            "config": self.config,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationRun':
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            test_suite=TestSuite.from_dict(data["test_suite"]),
            results=[EvaluationResult.from_dict(r) for r in data["results"]],
            config=data["config"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            metadata=data.get("metadata", {})
        )


@dataclass
class ComparisonResult:
    """
    Result of comparing multiple evaluation runs.
    
    Attributes:
        runs: List of evaluation runs being compared
        comparison_metrics: Metrics comparing the runs
        statistical_tests: Results of statistical significance tests
        best_run_by_metric: Best run for each metric
        metadata: Additional comparison metadata
    """
    runs: List[EvaluationRun]
    comparison_metrics: Dict[str, Dict[str, float]]  # run_id -> metric -> value
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    best_run_by_metric: Dict[str, str] = field(default_factory=dict)  # metric -> run_id
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_winner(self, metric: str = "f1_score") -> Optional[str]:
        """Get the best run ID for a given metric."""
        return self.best_run_by_metric.get(metric)
    
    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Get all metrics for a specific run."""
        return self.comparison_metrics.get(run_id, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "runs": [r.to_dict() for r in self.runs],
            "comparison_metrics": self.comparison_metrics,
            "statistical_tests": self.statistical_tests,
            "best_run_by_metric": self.best_run_by_metric,
            "metadata": self.metadata
        }
    
    def generate_summary(self) -> str:
        """Generate a text summary of the comparison."""
        lines = ["Evaluation Run Comparison Summary", "=" * 40]
        
        # List runs
        lines.append("\nRuns Compared:")
        for run in self.runs:
            lines.append(f"  - {run.name} (ID: {run.id})")
        
        # Best performers
        lines.append("\nBest Performers by Metric:")
        for metric, run_id in self.best_run_by_metric.items():
            run_name = next((r.name for r in self.runs if r.id == run_id), "Unknown")
            value = self.comparison_metrics.get(run_id, {}).get(metric, 0.0)
            lines.append(f"  - {metric}: {run_name} ({value:.3f})")
        
        # Statistical significance
        if self.statistical_tests:
            lines.append("\nStatistical Significance:")
            for test_name, results in self.statistical_tests.items():
                lines.append(f"  - {test_name}: {results}")
        
        return "\n".join(lines)