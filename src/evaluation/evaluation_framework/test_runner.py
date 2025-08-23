"""
Test execution engine for the evaluation framework.
Supports sequential, parallel, and batch execution modes.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import traceback
from datetime import datetime

from .models import TestCase, EvaluationResult, TestStatus
from .config import ExecutionConfig, TestExecutionMode, SearchStrategyConfig, NoiseConfig
from .data_loader import NoiseDataLoader, ToolBenchDataLoader, DataConfig

# Import services
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.search_service import SearchService, SearchStrategy
from src.services.embedding_enhancer import ToolEmbeddingEnhancer
from src.core.models import Tool, ToolFilterRequest
from src.core.config import get_settings


@dataclass
class TestContext:
    """
    Context for test execution containing all necessary services and configurations.
    """
    vector_store: VectorStoreService
    embedding_service: EmbeddingService
    search_service: SearchService
    search_config: SearchStrategyConfig
    noise_config: NoiseConfig
    noise_pool: Optional[List[Dict[str, Any]]] = None
    
    async def cleanup(self):
        """Cleanup resources after test execution."""
        if self.vector_store:
            await self.vector_store.clear_cache()


class TestExecutor(ABC):
    """
    Abstract base class for test executors.
    Defines the interface for different execution strategies.
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize executor with configuration.
        
        Args:
            config: Execution configuration
        """
        self.config = config
        self.progress_callback: Optional[Callable[[int, int], None]] = None
    
    @abstractmethod
    async def execute(
        self,
        test_cases: List[TestCase],
        test_context: TestContext
    ) -> List[EvaluationResult]:
        """
        Execute test cases.
        
        Args:
            test_cases: List of test cases to execute
            test_context: Test execution context
            
        Returns:
            List of evaluation results
        """
        pass
    
    def set_progress_callback(self, callback: Callable[[int, int], None]):
        """
        Set callback for progress updates.
        
        Args:
            callback: Function called with (completed, total) counts
        """
        self.progress_callback = callback
    
    async def _execute_single_test(
        self,
        test_case: TestCase,
        test_context: TestContext
    ) -> EvaluationResult:
        """
        Execute a single test case.
        
        Args:
            test_case: Test case to execute
            test_context: Execution context
            
        Returns:
            Evaluation result
        """
        start_time = time.time()
        
        try:
            # Clear collection if configured
            if hasattr(self.config, 'clear_collection_between_tests') and self.config.clear_collection_between_tests:
                try:
                    test_context.vector_store.client.delete_collection(test_context.vector_store.collection_name)
                    await test_context.vector_store.initialize()
                except:
                    pass
            
            # Index tools for this test case
            indexed_tools = await self._index_tools(
                test_case.available_tools,
                test_context
            )
            
            # Add noise tools if configured
            tools_with_noise = indexed_tools.copy()
            if test_context.noise_config.add_noise_to_available > 0 and test_context.noise_pool:
                import random
                
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
                    
                    # INDEX the noise tools in the vector store!
                    await self._index_tools(noise_tools, test_context)
                    
                    tools_with_noise.extend(noise_tools)
            
            # Create request
            request = ToolFilterRequest(
                messages=[{"role": "user", "content": test_case.query}],
                available_tools=self._convert_to_tools(tools_with_noise),
                max_tools=test_context.search_config.max_tools
            )
            
            # Perform search
            recommended_tools = await test_context.search_service.search(
                messages=request.messages,
                available_tools=request.available_tools,
                strategy=test_context.search_config.strategy,
                limit=request.max_tools,
                score_threshold=test_context.search_config.primary_similarity_threshold
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Calculate basic metrics (detailed metrics will be calculated by MetricsCalculator)
            expected_set = set(test_case.expected_tools)
            recommended_set = {
                tool.get("name", "")
                for tool in recommended_tools
                if tool.get("name")
            }
            
            true_positives = len(expected_set & recommended_set)
            false_positives = len(recommended_set - expected_set)
            false_negatives = len(expected_set - recommended_set)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return EvaluationResult(
                test_case=test_case,
                recommended_tools=recommended_tools,
                metrics={
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives
                },
                execution_time_ms=execution_time,
                status=TestStatus.COMPLETED,
                metadata={
                    "num_noise_tools": len(tools_with_noise) - len(indexed_tools),
                    "search_strategy": test_context.search_config.strategy.value
                }
            )
            
        except asyncio.TimeoutError:
            return EvaluationResult(
                test_case=test_case,
                recommended_tools=[],
                metrics={},
                execution_time_ms=(time.time() - start_time) * 1000,
                status=TestStatus.TIMEOUT,
                error="Test execution timed out"
            )
        except Exception as e:
            return EvaluationResult(
                test_case=test_case,
                recommended_tools=[],
                metrics={},
                execution_time_ms=(time.time() - start_time) * 1000,
                status=TestStatus.FAILED,
                error=f"{str(e)}\n{traceback.format_exc()}"
            )
    
    async def _index_tools(
        self,
        tools: List[Dict[str, Any]],
        test_context: TestContext
    ) -> List[Dict[str, Any]]:
        """
        Index tools in vector store.
        
        Args:
            tools: List of tool dictionaries
            test_context: Execution context
            
        Returns:
            Indexed tools
        """
        if not tools:
            return []
        
        # Generate embeddings
        enhancer = ToolEmbeddingEnhancer()
        tool_texts = []
        
        for tool_dict in tools:
            # Convert dict to Tool object for enhancer
            tool = Tool(**tool_dict)
            text = enhancer.tool_to_rich_text(tool)
            tool_texts.append(text)
        
        embeddings = await test_context.embedding_service.embed_batch(tool_texts)
        
        # Index in vector store
        await test_context.vector_store.index_tools_batch(tools, embeddings)
        
        return tools
    
    def _convert_to_tools(self, tool_dicts: List[Dict[str, Any]]) -> List[Tool]:
        """
        Convert tool dictionaries to Tool objects.
        
        Args:
            tool_dicts: List of tool dictionaries
            
        Returns:
            List of Tool objects
        """
        tools = []
        for tool_dict in tool_dicts:
            try:
                tools.append(Tool(**tool_dict))
            except Exception as e:
                print(f"Error converting tool: {e}")
        return tools


class SequentialExecutor(TestExecutor):
    """
    Execute tests sequentially, one after another.
    Good for debugging and when tests share resources.
    """
    
    async def execute(
        self,
        test_cases: List[TestCase],
        test_context: TestContext
    ) -> List[EvaluationResult]:
        """
        Execute test cases sequentially.
        
        Args:
            test_cases: List of test cases
            test_context: Execution context
            
        Returns:
            List of results in the same order as test cases
        """
        results = []
        total = len(test_cases)
        
        for idx, test_case in enumerate(test_cases):
            # Execute with retry logic if configured
            attempts = 0
            result = None
            
            while attempts < (self.config.max_retries if self.config.retry_on_failure else 1):
                attempts += 1
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        self._execute_single_test(test_case, test_context),
                        timeout=self.config.timeout_per_test
                    )
                    
                    # Break if successful
                    if result.status == TestStatus.COMPLETED:
                        break
                        
                except asyncio.TimeoutError:
                    result = EvaluationResult(
                        test_case=test_case,
                        recommended_tools=[],
                        metrics={},
                        execution_time_ms=self.config.timeout_per_test * 1000,
                        status=TestStatus.TIMEOUT,
                        error=f"Timeout after {self.config.timeout_per_test}s"
                    )
                    
                # Add delay before retry
                if attempts < self.config.max_retries and self.config.retry_on_failure:
                    await asyncio.sleep(1)
            
            results.append(result)
            
            # Update progress
            if self.progress_callback:
                self.progress_callback(idx + 1, total)
        
        return results


class ParallelExecutor(TestExecutor):
    """
    Execute tests in parallel using asyncio.
    Good for independent tests and faster execution.
    """
    
    async def execute(
        self,
        test_cases: List[TestCase],
        test_context: TestContext
    ) -> List[EvaluationResult]:
        """
        Execute test cases in parallel.
        
        Args:
            test_cases: List of test cases
            test_context: Execution context
            
        Returns:
            List of results (order may differ from input)
        """
        total = len(test_cases)
        completed = 0
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.config.num_workers)
        
        async def execute_with_semaphore(test_case: TestCase) -> EvaluationResult:
            async with semaphore:
                result = await asyncio.wait_for(
                    self._execute_single_test(test_case, test_context),
                    timeout=self.config.timeout_per_test
                )
                
                nonlocal completed
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, total)
                
                return result
        
        # Execute all tests in parallel
        tasks = [
            execute_with_semaphore(test_case)
            for test_case in test_cases
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    EvaluationResult(
                        test_case=test_cases[idx],
                        recommended_tools=[],
                        metrics={},
                        execution_time_ms=0,
                        status=TestStatus.FAILED,
                        error=str(result)
                    )
                )
            else:
                processed_results.append(result)
        
        return processed_results


class BatchExecutor(TestExecutor):
    """
    Execute tests in batches.
    Good for balancing parallelism and resource usage.
    """
    
    async def execute(
        self,
        test_cases: List[TestCase],
        test_context: TestContext
    ) -> List[EvaluationResult]:
        """
        Execute test cases in batches.
        
        Args:
            test_cases: List of test cases
            test_context: Execution context
            
        Returns:
            List of results
        """
        results = []
        total = len(test_cases)
        completed = 0
        
        # Process in batches
        for i in range(0, len(test_cases), self.config.batch_size):
            batch = test_cases[i:i + self.config.batch_size]
            
            # Execute batch in parallel
            batch_tasks = [
                asyncio.wait_for(
                    self._execute_single_test(test_case, test_context),
                    timeout=self.config.timeout_per_test
                )
                for test_case in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for idx, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    results.append(
                        EvaluationResult(
                            test_case=batch[idx],
                            recommended_tools=[],
                            metrics={},
                            execution_time_ms=0,
                            status=TestStatus.FAILED,
                            error=str(result)
                        )
                    )
                else:
                    results.append(result)
                
                completed += 1
                if self.progress_callback:
                    self.progress_callback(completed, total)
        
        return results


class TestRunner:
    """
    Main test runner that orchestrates test execution.
    """
    
    def __init__(
        self,
        execution_config: ExecutionConfig,
        search_config: SearchStrategyConfig,
        noise_config: NoiseConfig
    ):
        """
        Initialize test runner.
        
        Args:
            execution_config: Execution configuration
            search_config: Search strategy configuration
            noise_config: Noise configuration
        """
        self.execution_config = execution_config
        self.search_config = search_config
        self.noise_config = noise_config
        
        # Create executor based on mode
        self.executor = self._create_executor()
        
        # Services will be initialized when running
        self.test_context: Optional[TestContext] = None
    
    def _create_executor(self) -> TestExecutor:
        """
        Create executor based on execution mode.
        
        Returns:
            Appropriate executor instance
        """
        mode_to_executor = {
            TestExecutionMode.SEQUENTIAL: SequentialExecutor,
            TestExecutionMode.PARALLEL: ParallelExecutor,
            TestExecutionMode.BATCH: BatchExecutor
        }
        
        executor_class = mode_to_executor.get(
            self.execution_config.mode,
            SequentialExecutor
        )
        
        return executor_class(self.execution_config)
    
    async def initialize_services(self) -> TestContext:
        """
        Initialize required services.
        
        Returns:
            Test context with initialized services
        """
        settings = get_settings()
        
        # Initialize embedding service
        embedding_service = EmbeddingService(
            model=self.search_config.primary_embedding_model,
            api_key=settings.primary_embedding_api_key
        )
        
        # Initialize vector store
        dimension = settings.get_embedding_dimension(
            self.search_config.primary_embedding_model
        )
        vector_store = VectorStoreService(
            embedding_dimension=dimension,
            model_name=self.search_config.primary_embedding_model,
            similarity_threshold=self.search_config.primary_similarity_threshold
        )
        await vector_store.initialize()
        
        # Initialize search service
        search_service = SearchService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        # Load noise pool if configured
        noise_pool = None
        if self.noise_config.add_noise_to_available > 0:
            # Create noise loader using data path from noise config
            data_config = DataConfig(
                data_source=self.noise_config.noise_data_source,
                data_path=self.noise_config.noise_data_path
            )
            base_loader = ToolBenchDataLoader(data_config)
            noise_loader = NoiseDataLoader(data_config, base_loader)
            
            # Load noise pool
            noise_pool = noise_loader.load_noise_pool(
                target_size=self.noise_config.noise_pool_size
            )
        
        return TestContext(
            vector_store=vector_store,
            embedding_service=embedding_service,
            search_service=search_service,
            search_config=self.search_config,
            noise_config=self.noise_config,
            noise_pool=noise_pool
        )
    
    async def run(
        self,
        test_cases: List[TestCase],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[EvaluationResult]:
        """
        Run test cases.
        
        Args:
            test_cases: List of test cases to run
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of evaluation results
        """
        # Initialize services if not already done
        if self.test_context is None:
            self.test_context = await self.initialize_services()
        
        # Set progress callback
        if progress_callback:
            self.executor.set_progress_callback(progress_callback)
        
        # Execute tests
        results = await self.executor.execute(test_cases, self.test_context)
        
        return results
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.test_context:
            await self.test_context.cleanup()
            self.test_context = None