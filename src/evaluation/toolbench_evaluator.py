"""
ToolBench Evaluator - Transform and evaluate ToolBench data with our API.
Uses test.json and train.json from toolbench_data/data/retrieval/G1/
"""

import json
import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from src.core.models import Tool, ToolFilterRequest, ChatMessage, ToolFunction
from src.services.embedding_enhancer import ToolEmbeddingEnhancer
from src.services.query_enhancer import QueryEnhancer
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.search_service import SearchService, SearchStrategy
from src.core.config import get_settings
from src.evaluation.threshold_optimizer import ThresholdOptimizer
from slugify import slugify

# Configure logging to show DEBUG messages
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG to see all debug messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ToolBenchEvaluator:
    """Evaluate our tool filtering API using ToolBench data."""

    def __init__(self, data_path: str = "toolbench_data/data/retrieval/G1"):
        self.data_path = Path(data_path)
        self.settings = get_settings()
        # Cache for noise tools to avoid reloading
        self._cached_noise_tools = None
        self._cached_noise_embeddings = None

    def convert_api_to_openai_format(self, api: Dict[str, Any]) -> Tool:
        """
        Convert a single ToolBench API to OpenAI function format.

        Args:
            api: ToolBench API definition

        Returns:
            Tool object in OpenAI format
        """
        # Build properties from parameters
        properties = {}
        required = []

        # Process required parameters
        for param in api.get("required_parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "string").lower()

            # Map ToolBench types to JSON Schema types
            if param_type == "string":
                json_type = "string"
            elif param_type in ["number", "integer", "int"]:
                json_type = "number"
            elif param_type == "boolean":
                json_type = "boolean"
            elif param_type == "enum":
                json_type = "string"
            else:
                json_type = "string"

            properties[param_name] = {
                "type": json_type,
                "description": param.get("description", "")
            }

            if param.get("default"):
                properties[param_name]["default"] = param["default"]

            required.append(param_name)

        # Process optional parameters
        for param in api.get("optional_parameters", []):
            param_name = param["name"]
            param_type = param.get("type", "string").lower()

            # Map types
            if param_type == "string":
                json_type = "string"
            elif param_type in ["number", "integer", "int"]:
                json_type = "number"
            elif param_type == "boolean":
                json_type = "boolean"
            elif param_type == "enum":
                json_type = "string"
            else:
                json_type = "string"

            properties[param_name] = {
                "type": json_type,
                "description": param.get("description", "")
            }

            if param.get("default"):
                properties[param_name]["default"] = param["default"]

        # Create function name combining tool and API name
        tool_name = api.get("tool_name", "Unknown")
        api_name = api.get("api_name", "Unknown")
        # Create the raw function name first
        raw_function_name = f"{tool_name}_{api_name}"
        # Normalize it using slugify to handle special characters consistently
        function_name = slugify(raw_function_name, separator="_")

        # Create the Tool object

        return Tool(
            type="function",
            function=ToolFunction(
                name=function_name,
                description=api.get("api_description", ""),
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            )
        )

    def load_test_data(self, filename: str = "G1_instruction.json", data_dir: str = "toolbench_data/data/test_instruction") -> List[Dict[str, Any]]:
        """Load test data from ToolBench JSON file."""
        file_path = Path(data_dir) / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Test file not found: {file_path}")

        with open(file_path, 'r') as f:
            return json.load(f)

    async def index_tools_from_case(
        self,
        test_case: Dict[str, Any],
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService
    ) -> List[Tool]:
        """
        Index all tools from a test case into the vector store.

        Args:
            test_case: ToolBench test case with api_list
            vector_store: Vector store service
            embedding_service: Embedding service

        Returns:
            List of converted Tool objects
        """
        api_list = test_case.get("api_list", [])

        # Convert APIs to Tool format
        tools = []
        for api in api_list:
            try:
                tool = self.convert_api_to_openai_format(api)
                tools.append(tool)
            except Exception as e:
                print(f"Error converting API {api.get('api_name', 'Unknown')}: {e}")
                continue

        if not tools:
            return []

        # Generate embeddings for tools
        tool_texts = []
        enhancer = ToolEmbeddingEnhancer()
        for tool in tools:
            # Combine function name and description for embedding
            func = tool.function
            text = enhancer.tool_to_rich_text(tool)
            tool_texts.append(text)

        # Batch generate embeddings
        embeddings = await embedding_service.embed_batch(tool_texts)

        # Convert Tool objects to dict format for indexing
        tool_dicts = []
        for tool in tools:
            tool_dicts.append(tool.model_dump())

        # Index tools with embeddings
        print(f"  Indexing {len(tool_dicts)} tools with {len(embeddings)} embeddings")
        print(f"  First tool: {tools[0].function.name if tools else 'None'}")
        print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")

        # Debug: Print the actual tool names being indexed
        indexed_names = [tool.function.name for tool in tools]
        print(f"  Tool names being indexed: {indexed_names}")

        await vector_store.index_tools_batch(tool_dicts, embeddings)

        # Verify indexing by checking collection info
        try:
            collection_info = vector_store.client.get_collection(vector_store.collection_name)
            print(f"  Collection '{vector_store.collection_name}' has {collection_info.points_count} points")

            # Optimize collection if it's large enough
            if collection_info.points_count > self.settings.two_stage_threshold:
                print(f"  Optimizing collection for {collection_info.points_count} tools...")
                await vector_store.optimize_collection(target_mode="balanced")

            # Debug: Verify the tools are searchable by doing a test query
            if tools:
                test_tool_name = tools[0].function.name
                print(f"  Verifying tool '{test_tool_name}' is searchable...")

                # Try to retrieve the specific tool
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                test_filter = Filter(must=[
                    FieldCondition(
                        key="name",
                        match=MatchValue(value=test_tool_name)
                    )
                ])

                test_results = vector_store.client.scroll(
                    collection_name=vector_store.collection_name,
                    scroll_filter=test_filter,
                    limit=1
                )

                if test_results[0]:
                    print(f"    ✓ Tool found in collection")
                else:
                    print(f"    ✗ Tool NOT found in collection - indexing may have failed")

        except Exception as e:
            print(f"  Error checking collection: {e}")

        return tools

    def evaluate_results(
        self,
        query: str,
        expected_apis: List[List[str]],
        recommended_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate recommendation results against expected APIs.

        Args:
            query: The user query
            expected_apis: List of [tool_name, api_name] pairs that are expected
            recommended_tools: Tools recommended by our API

        Returns:
            Evaluation metrics
        """
        # Create expected set from relevant APIs (normalized using slugify)
        expected_set = set()
        for api_ref in expected_apis:
            if len(api_ref) >= 2:
                tool_name = api_ref[0]
                api_name = api_ref[1]
                raw_name = f"{tool_name}_{api_name}"
                # Use slugify to normalize the same way as when creating tools
                expected_name = slugify(raw_name, separator="_")
                expected_set.add(expected_name)

        # Extract recommended tool names
        recommended_set = set()
        for tool in recommended_tools:
            tool_name = tool.get("tool_name", "")
            if not tool_name and "original" in tool:
                # Try to extract from original tool format
                original = tool["original"]
                if isinstance(original, dict) and "function" in original:
                    func = original["function"]
                    if isinstance(func, dict):
                        tool_name = func.get("name", "")
                    else:
                        # It's a ToolFunction object
                        tool_name = func.name if hasattr(func, 'name') else ""

            if tool_name:
                # No need to lowercase since slugify already does that
                recommended_set.add(tool_name)

        # Calculate metrics
        if not expected_set:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "no_ground_truth": True
            }

        true_positives = len(expected_set & recommended_set)
        false_positives = len(recommended_set - expected_set)
        false_negatives = len(expected_set - recommended_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "expected_tools": list(expected_set),
            "recommended_tools": list(recommended_set)
        }

    async def run_single_evaluation(
        self,
        test_case: Dict[str, Any],
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        search_service: SearchService,
        add_noise_to_available: int = 0,
        noise_tools_pool: List[Tool] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation for a single test case.

        Args:
            test_case: ToolBench test case
            vector_store: Vector store service
            embedding_service: Embedding service
            search_service: Search service for orchestrating search strategies
            add_noise_to_available: Number of noise tools to add to available_tools
            noise_tools_pool: Pool of noise tools to sample from

        Returns:
            Evaluation results for this test case
        """
        query = test_case.get("query", "")
        query_id = test_case.get("query_id", 0)
        expected_apis = test_case.get("relevant APIs", [])

        print(f"\nQuery {query_id}: {query[:100]}...")

        # Index tools from this test case
        start_time = time.time()
        tools = await self.index_tools_from_case(test_case, vector_store, embedding_service)
        index_time = time.time() - start_time

        print(f"  Indexed {len(tools)} tools in {index_time:.2f}s")

        # Add noise tools to available_tools if requested
        original_tool_count = len(tools)
        noise_tool_names = []
        if add_noise_to_available > 0 and noise_tools_pool:
            import random
            # Get tool names from test case to avoid duplicates
            test_tool_names = {tool.function.name for tool in tools}

            # Filter noise pool to exclude tools already in test case
            available_noise = [
                tool for tool in noise_tools_pool
                if tool.function.name not in test_tool_names
            ]

            # Sample noise tools
            if len(available_noise) >= add_noise_to_available:
                noise_tools = random.sample(available_noise, add_noise_to_available)
            else:
                noise_tools = available_noise
                print(f"  WARNING: Only {len(available_noise)} noise tools available, requested {add_noise_to_available}")

            # Track noise tool names for analysis
            noise_tool_names = [tool.function.name for tool in noise_tools]

            # Add noise tools to the available tools list
            tools.extend(noise_tools)
            print(f"  Added {len(noise_tools)} noise tools to available_tools (total: {len(tools)})")
            print(f"  Noise ratio: {len(noise_tools)/len(tools)*100:.1f}% of available tools are noise")

        # Create request with the query
        request = ToolFilterRequest(
            messages=[{"role": "user", "content": query}],
            available_tools=tools,  # Pass the indexed tools + noise tools
            max_tools=10
        )

        # Extract tool names for filtering (only search within indexed tools for this test case)
        available_tool_names = []
        for tool in tools:
            available_tool_names.append(tool.function.name)

        print(f"  Available tool names for filter: {len(available_tool_names)} tools")

        # Determine search strategy based on configuration
        if getattr(self.settings, 'enable_cross_encoder', True) and getattr(self.settings, 'enable_hybrid_search', True):
            strategy = SearchStrategy.HYBRID_CROSS_ENCODER
            print(f"  Using strategy: Hybrid + Cross-Encoder Reranking")
        elif getattr(self.settings, 'enable_hybrid_search', True):
            strategy = SearchStrategy.HYBRID
            print(f"  Using strategy: Hybrid (semantic + BM25)")
        elif getattr(self.settings, 'enable_query_enhancement', True):
            strategy = SearchStrategy.SEMANTIC
            print(f"  Using strategy: Semantic with query enhancement")
        else:
            strategy = SearchStrategy.SEMANTIC
            print(f"  Using strategy: Semantic")

        start_time = time.time()

        # For query enhancement, we need special handling
        if strategy == SearchStrategy.SEMANTIC and getattr(self.settings, 'enable_query_enhancement', True):
            # Enhanced multi-query search
            query_enhancer = QueryEnhancer()
            enhanced_query = query_enhancer.enhance_query(request.messages, [])

            # Generate embeddings for all query representations
            query_embeddings = await embedding_service.embed_queries(enhanced_query["queries"])
            embed_time = time.time() - start_time

            print(f"  Enhanced query with {len(query_embeddings)} representations")
            print(f"  Query aspects: {list(query_embeddings.keys())}")

            # Use search_multi_query for enhanced search
            all_results = await search_service.search_multi_query(
                query_embeddings=query_embeddings,
                weights=enhanced_query["weights"],
                limit=100,  # High limit to ensure we get all available tools
                score_threshold=0.0,  # No threshold to see all scores
                filter_dict={"name": available_tool_names}  # Only search within tools for this test case
            )
        else:
            # Use unified search interface for all other strategies
            all_results = await search_service.search(
                messages=request.messages,
                available_tools=tools,
                strategy=strategy,
                limit=100,
                score_threshold=0.0  # Get all scores for threshold optimization
            )
            embed_time = time.time() - start_time

        print(f"  Available tools: {len(available_tool_names)}")
        print(f"  All results count: {len(all_results)}")

        # Sanity check: with zero threshold and filtering, we should get back exactly the available tools
        if len(all_results) != len(available_tool_names):
            print(f"  WARNING: Expected {len(available_tool_names)} results but got {len(all_results)}")
            print(f"  Missing tools might not be indexed properly!")

        print(f"  All scores: {[r.get('score', 0) for r in all_results]}")

        # Print tool names for debugging
        if all_results:
            print(f"  Top tools found:")
            for i, result in enumerate(all_results[:3]):
                print(f"    {i+1}. {result.get('tool_name', 'Unknown')} (score: {result.get('score', 0):.3f})")

        # Now search with threshold - use a lower threshold for testing
        test_threshold = self.settings.primary_similarity_threshold  # Lower threshold to get some results

        # Perform the same search but with threshold applied
        if strategy == SearchStrategy.SEMANTIC and getattr(self.settings, 'enable_query_enhancement', True):
            # Enhanced search with threshold
            recommended = await search_service.search_multi_query(
                query_embeddings=query_embeddings,
                weights=enhanced_query["weights"],
                limit=request.max_tools,
                score_threshold=test_threshold,
                filter_dict={"name": available_tool_names}  # Only search within tools for this test case
            )
        else:
            # Use unified search interface with threshold
            recommended = await search_service.search(
                messages=request.messages,
                available_tools=tools,
                strategy=strategy,
                limit=request.max_tools,
                score_threshold=test_threshold
            )

        search_time = time.time() - start_time

        print(f"  Found {len(recommended)} recommendations with threshold {test_threshold}")

        # Evaluate results
        metrics = self.evaluate_results(query, expected_apis, recommended)

        # Print expected vs retrieved for visual inspection
        print(f"  Expected tools: {metrics.get('expected_tools', [])}")
        print(f"  Retrieved tools: {metrics.get('recommended_tools', [])}")

        # Add timing information
        metrics.update({
            "query_id": query_id,
            "query": query,
            "num_tools_indexed": original_tool_count,  # Original tools without noise
            "num_tools_available": len(tools),  # Total including noise
            "num_noise_tools_added": len(noise_tool_names),
            "num_tools_recommended": len(recommended),
            "index_time_ms": index_time * 1000,
            "embed_time_ms": embed_time * 1000,
            "search_time_ms": search_time * 1000,
            "total_time_ms": (index_time + embed_time + search_time) * 1000,
            "all_scores": all_results  # Add all scores for threshold optimization
        })

        # Analyze noise impact on results
        if noise_tool_names:
            # Track which recommended tools are noise vs expected
            recommended_noise = []
            recommended_expected = []
            expected_ranks = []

            for idx, tool in enumerate(recommended):
                tool_name = tool.get("tool_name", "")
                if tool_name in noise_tool_names:
                    recommended_noise.append(tool_name)
                elif tool_name in metrics.get('expected_tools', []):
                    recommended_expected.append(tool_name)
                    expected_ranks.append(idx + 1)  # 1-indexed rank

            # Calculate metrics that matter
            metrics["noise_tools_in_results"] = recommended_noise
            metrics["num_noise_in_results"] = len(recommended_noise)
            metrics["expected_tools_in_results"] = recommended_expected
            metrics["num_expected_in_results"] = len(recommended_expected)

            # Proportion of results that are noise (not necessarily bad)
            metrics["noise_proportion"] = len(recommended_noise) / len(recommended) if recommended else 0

            # Critical: Are we missing any expected tools?
            missing_expected = [t for t in metrics.get('expected_tools', []) if t not in recommended_expected]
            metrics["missing_expected_tools"] = missing_expected
            metrics["expected_tool_recall"] = len(recommended_expected) / len(metrics.get('expected_tools', [])) if metrics.get('expected_tools') else 0

            # Ranking quality: Average rank of expected tools
            if expected_ranks:
                metrics["avg_rank_expected_tools"] = sum(expected_ranks) / len(expected_ranks)
                metrics["best_rank_expected_tool"] = min(expected_ranks)
            else:
                metrics["avg_rank_expected_tools"] = float('inf')
                metrics["best_rank_expected_tool"] = float('inf')

            # Print analysis
            if missing_expected:
                print(f"  ❌ CRITICAL: Missing expected tools: {missing_expected}")
                print(f"  Expected tool recall: {metrics['expected_tool_recall']*100:.1f}%")
            else:
                print(f"  ✅ All expected tools found in results!")

            if expected_ranks:
                print(f"  Expected tools ranking: positions {expected_ranks}, avg={metrics['avg_rank_expected_tools']:.1f}")

            print(f"  Noise proportion in results: {metrics['noise_proportion']*100:.1f}% ({len(recommended_noise)}/{len(recommended)} tools)")

        print(f"  Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
        if metrics.get('true_positives', 0) > 0:
            print(f"  Matches found: {metrics['true_positives']} tools")
        print(f"  Time: {metrics['total_time_ms']:.1f}ms")

        return metrics

    async def load_random_toolbench_tools(self,
                                          data_dir: str = "toolbench_data/data/test_instruction",
                                          target_total: int = 1500) -> List[Tool]:
        """
        Load a random sample of tools from ToolBench test instruction files.

        Args:
            data_dir: Directory containing test instruction files
            target_total: Target number of tools to load (default 1500)

        Returns:
            List of randomly selected Tool objects
        """
        import random

        all_tools = []
        seen_tool_names = set()

        # List of available test files
        test_files = ["G1_instruction.json", "G2_instruction.json", "G3_instruction.json", "G1_category.json", "G1_tool.json"]

        # Randomize the order of files
        random.seed(42)  # Fixed seed for reproducibility
        random.shuffle(test_files)

        for test_file in test_files:
            if len(all_tools) >= target_total:
                break

            try:
                file_path = Path(data_dir) / test_file
                if not file_path.exists():
                    print(f"  Skipping {test_file} - file not found")
                    continue

                print(f"  Loading tools from {test_file}...")

                with open(file_path, 'r') as f:
                    test_data = json.load(f)

                # Collect all unique tools from this file
                file_tools = []
                for test_case in test_data:
                    api_list = test_case.get("api_list", [])

                    for api in api_list:
                        try:
                            tool = self.convert_api_to_openai_format(api)
                            # Avoid duplicates based on tool name
                            if tool.function.name not in seen_tool_names:
                                file_tools.append(tool)
                                seen_tool_names.add(tool.function.name)
                        except Exception as e:
                            # Skip tools that fail to convert
                            continue

                # Determine how many tools to take from this file
                remaining_needed = target_total - len(all_tools)

                if len(file_tools) > 0:
                    # Take between 400-600 tools from each file, or whatever is available/needed
                    min_from_file = min(400, len(file_tools), remaining_needed)
                    max_from_file = min(600, len(file_tools), remaining_needed)

                    # Randomly decide how many to take within the range
                    num_to_take = random.randint(min_from_file, max_from_file)

                    # Randomly sample tools from this file
                    selected_from_file = random.sample(file_tools, num_to_take)
                    all_tools.extend(selected_from_file)

                    print(f"    Selected {num_to_take} tools from {len(file_tools)} available in {test_file}")
                else:
                    print(f"    No valid tools found in {test_file}")

            except Exception as e:
                print(f"  Error loading {test_file}: {e}")

        print(f"  Total tools loaded: {len(all_tools)} (target was {target_total})")
        return all_tools

    async def create_sample_noise_tools(
        self,
        vector_store: VectorStoreService,
        embedding_service: EmbeddingService,
        num_tools: int = 1500
    ):
        """
        Index real ToolBench tools as noise to simulate a realistic environment.
        Uses cached tools if available to avoid reloading and re-embedding.

        Args:
            vector_store: Vector store service
            embedding_service: Embedding service
            num_tools: Number of noise tools to index (default 1500)
        """
        # Check if we have cached tools
        if self._cached_noise_tools is None or self._cached_noise_embeddings is None:
            print(f"Loading and preparing {num_tools} real ToolBench tools as noise...")

            # Load random sample of tools directly
            selected_tools = await self.load_random_toolbench_tools(target_total=num_tools)

            if not selected_tools:
                print("  WARNING: No tools loaded from ToolBench data, skipping noise tools")
                return

            # Generate embeddings for selected tools
            tool_texts = []
            enhancer = ToolEmbeddingEnhancer()

            for tool in selected_tools:
                text = enhancer.tool_to_rich_text(tool)
                tool_texts.append(text)

            print(f"  Generating embeddings for {len(selected_tools)} noise tools...")
            embeddings = await embedding_service.embed_batch(tool_texts)

            # Convert Tool objects to dict format
            tool_dicts = []
            for tool in selected_tools:
                tool_dicts.append(tool.model_dump())

            # Cache for future use
            self._cached_noise_tools = tool_dicts
            self._cached_noise_embeddings = embeddings

            print(f"  Cached {len(tool_dicts)} noise tools with embeddings")
        else:
            print(f"  Using cached {len(self._cached_noise_tools)} noise tools")

        # Index the cached tools
        print(f"  Indexing {len(self._cached_noise_tools)} noise tools...")
        await vector_store.index_tools_batch(
            self._cached_noise_tools,
            self._cached_noise_embeddings
        )

        # Verify indexing
        try:
            collection_info = vector_store.client.get_collection(vector_store.collection_name)
            print(f"  Collection now has {collection_info.points_count} total tools")
        except Exception as e:
            print(f"  Error checking collection: {e}")

    async def run_evaluation(
        self,
        test_file: str = "G1_instruction.json",
        data_dir: str = "toolbench_data/data/test_instruction",
        num_cases: int = 50,
        clear_collection: bool = True,
        add_noise_tools: bool = True,
        add_noise_to_available: int = 0
    ) -> Dict[str, Any]:
        """
        Run evaluation on ToolBench test data.

        Args:
            test_file: Name of test file to use
            data_dir: Directory containing test data
            num_cases: Number of test cases to evaluate
            clear_collection: Whether to clear the collection before each test
            add_noise_tools: Whether to add sample noise tools to vector store
            add_noise_to_available: Number of noise tools to add to available_tools per test

        Returns:
            Evaluation summary and results
        """
        # Load test data
        test_data = self.load_test_data(test_file, data_dir)
        test_cases = test_data[:num_cases]

        print(f"Loaded {len(test_cases)} test cases from {test_file}")

        # Initialize services
        embedding_service = EmbeddingService(
            model=self.settings.primary_embedding_model,
            api_key=self.settings.primary_embedding_api_key
        )

        dimension = self.settings.get_embedding_dimension(self.settings.primary_embedding_model)
        vector_store = VectorStoreService(
            embedding_dimension=dimension,
            model_name=self.settings.primary_embedding_model,
            similarity_threshold=self.settings.primary_similarity_threshold
        )

        await vector_store.initialize()

        # Initialize search service
        search_service = SearchService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        print(f"Initialized search service with strategies: {[s.value for s in search_service.get_available_strategies()]}")

        # Clear cache to ensure consistent results
        await vector_store.clear_cache()
        print("Cleared search cache for consistent evaluation")

        # Add sample noise tools to simulate realistic environment
        if add_noise_tools:
            await self.create_sample_noise_tools(vector_store, embedding_service, num_tools=1500)

        # Load pool of noise tools for available_tools if needed
        noise_tools_pool = []
        if add_noise_to_available > 0:
            print(f"\nLoading pool of noise tools for available_tools testing...")
            # Load a larger pool of tools to sample from
            noise_tools_pool = await self.load_random_toolbench_tools(
                data_dir=data_dir,
                target_total=add_noise_to_available * 3  # Load 3x the needed amount for variety
            )
            print(f"  Loaded {len(noise_tools_pool)} tools for noise pool")

        # Initialize threshold optimizer
        threshold_optimizer = ThresholdOptimizer()

        # Run evaluations
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"Test Case {i+1}/{len(test_cases)}")
            print('='*60)

            # Clear collection if requested (to avoid cross-contamination)
            if clear_collection and i > 0:
                # Clear the collection by deleting and recreating it
                try:
                    vector_store.client.delete_collection(vector_store.collection_name)
                    await vector_store.initialize()

                    # Re-add noise tools after clearing
                    if add_noise_tools:
                        await self.create_sample_noise_tools(vector_store, embedding_service, num_tools=1500)
                except Exception as e:
                    print(f"  Warning: Could not clear collection: {e}")

            result = await self.run_single_evaluation(
                test_case,
                vector_store,
                embedding_service,
                search_service,
                add_noise_to_available=add_noise_to_available,
                noise_tools_pool=noise_tools_pool
            )
            results.append(result)

            # Add scores to threshold optimizer
            if 'all_scores' in result and 'expected_tools' in result:
                threshold_optimizer.add_scores_from_evaluation(
                    result['all_scores'],
                    result['expected_tools'],
                    result.get('query_id')
                )

        # Calculate aggregate metrics
        total_precision = sum(r["precision"] for r in results)
        total_recall = sum(r["recall"] for r in results)
        total_f1 = sum(r["f1_score"] for r in results)
        total_time = sum(r["total_time_ms"] for r in results)

        num_results = len(results)
        avg_precision = total_precision / num_results
        avg_recall = total_recall / num_results
        avg_f1 = total_f1 / num_results
        avg_time = total_time / num_results

        # Calculate per-case statistics
        precisions = [r["precision"] for r in results]
        recalls = [r["recall"] for r in results]
        f1_scores = [r["f1_score"] for r in results]

        # Calculate noise impact metrics if applicable
        noise_metrics = {}
        if add_noise_to_available > 0:
            # Focus on what matters: expected tool recall and ranking
            expected_recalls = [r.get("expected_tool_recall", 1.0) for r in results]
            avg_ranks = [r.get("avg_rank_expected_tools", float('inf')) for r in results
                        if r.get("avg_rank_expected_tools") != float('inf')]
            noise_proportions = [r.get("noise_proportion", 0) for r in results]
            missing_expected_counts = [len(r.get("missing_expected_tools", [])) for r in results]

            noise_metrics = {
                "noise_tools_per_test": add_noise_to_available,
                # Critical metrics
                "avg_expected_tool_recall": sum(expected_recalls) / len(expected_recalls) if expected_recalls else 0,
                "min_expected_tool_recall": min(expected_recalls) if expected_recalls else 0,
                "cases_with_perfect_recall": sum(1 for r in expected_recalls if r == 1.0),
                "cases_missing_expected": sum(1 for m in missing_expected_counts if m > 0),
                "total_missing_expected": sum(missing_expected_counts),
                # Ranking metrics
                "avg_rank_of_expected": sum(avg_ranks) / len(avg_ranks) if avg_ranks else float('inf'),
                # Noise proportion (informational)
                "avg_noise_proportion": sum(noise_proportions) / len(noise_proportions) if noise_proportions else 0,
                "max_noise_proportion": max(noise_proportions) if noise_proportions else 0
            }

        summary = {
            "test_file": test_file,
            "num_test_cases": num_results,
            "model": self.settings.primary_embedding_model,
            "similarity_threshold": self.settings.primary_similarity_threshold,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1_score": avg_f1,
                "min_precision": min(precisions),
                "max_precision": max(precisions),
                "min_recall": min(recalls),
                "max_recall": max(recalls),
                "min_f1": min(f1_scores),
                "max_f1": max(f1_scores)
            },
            "timing": {
                "avg_total_time_ms": avg_time,
                "total_time_ms": total_time
            }
        }

        # Add noise metrics if applicable
        if noise_metrics:
            summary["noise_impact"] = noise_metrics

        # Get performance stats if available
        try:
            performance_stats = await vector_store.get_performance_stats()
            summary["performance"] = performance_stats
        except:
            pass

        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Test Cases: {num_results}")
        print(f"Model: {self.settings.primary_embedding_model}")
        print(f"Threshold: {self.settings.primary_similarity_threshold}")
        print(f"\nMetrics:")
        print(f"  Average Precision: {avg_precision:.3f}")
        print(f"  Average Recall: {avg_recall:.3f}")
        print(f"  Average F1 Score: {avg_f1:.3f}")

        # Print noise impact if applicable
        if noise_metrics:
            print(f"\nNoise Impact Analysis:")
            print(f"  Noise tools added per test: {noise_metrics['noise_tools_per_test']}")

            # Critical: Expected tool recall
            print(f"\n  🎯 Expected Tool Recall (CRITICAL):")
            print(f"    Average recall: {noise_metrics['avg_expected_tool_recall']*100:.1f}%")
            print(f"    Minimum recall: {noise_metrics['min_expected_tool_recall']*100:.1f}%")
            print(f"    Cases with perfect recall: {noise_metrics['cases_with_perfect_recall']}/{num_results}")
            print(f"    Cases missing expected tools: {noise_metrics['cases_missing_expected']}/{num_results}")

            if noise_metrics['total_missing_expected'] > 0:
                print(f"    ❌ Total missing expected tools: {noise_metrics['total_missing_expected']}")

            # Ranking quality
            if noise_metrics['avg_rank_of_expected'] != float('inf'):
                print(f"\n  📊 Expected Tool Ranking:")
                print(f"    Average rank position: {noise_metrics['avg_rank_of_expected']:.1f}")

            # Noise proportion (informational)
            print(f"\n  📈 Noise in Results (informational):")
            print(f"    Average noise proportion: {noise_metrics['avg_noise_proportion']*100:.1f}%")
            print(f"    Maximum noise proportion: {noise_metrics['max_noise_proportion']*100:.1f}%")

            # Overall assessment based on recall
            print(f"\n  Overall Noise Resistance:")
            if noise_metrics['avg_expected_tool_recall'] >= 0.95:
                print(f"    ✅ EXCELLENT: {noise_metrics['avg_expected_tool_recall']*100:.1f}% expected tool recall")
            elif noise_metrics['avg_expected_tool_recall'] >= 0.85:
                print(f"    ✓ GOOD: {noise_metrics['avg_expected_tool_recall']*100:.1f}% expected tool recall")
            elif noise_metrics['avg_expected_tool_recall'] >= 0.75:
                print(f"    ⚠️  MODERATE: {noise_metrics['avg_expected_tool_recall']*100:.1f}% expected tool recall")
            else:
                print(f"    ❌ POOR: {noise_metrics['avg_expected_tool_recall']*100:.1f}% expected tool recall - needs improvement!")

        print(f"\nTiming:")
        print(f"  Average Time per Query: {avg_time:.1f}ms")
        print(f"  Total Time: {total_time/1000:.1f}s")

        # Print performance stats if available
        if "performance" in summary:
            perf = summary["performance"]
            print(f"\nOptimizer Performance:")
            if "optimizer" in perf and perf["optimizer"]:
                for op, stats in perf["optimizer"].items():
                    if isinstance(stats, dict) and "avg_ms" in stats:
                        print(f"  {op}: avg={stats['avg_ms']:.1f}ms, p95={stats.get('p95_ms', 0):.1f}ms")

            if "cache" in perf and perf["cache"]:
                cache = perf["cache"]
                print(f"\nCache Performance:")
                print(f"  Hit Rate: {cache.get('hit_rate', 0):.1%}")
                print(f"  Cache Size: {cache.get('size', 0)}/{cache.get('max_size', 0)}")
                print(f"  Total Hits: {cache.get('hits', 0)}, Misses: {cache.get('misses', 0)}")

        # Run threshold optimization analysis
        print("\n" + "="*60)
        print("THRESHOLD OPTIMIZATION ANALYSIS")
        print("="*60)

        optimal_thresholds = threshold_optimizer.find_optimal_threshold()

        print("\nScore Distribution:")
        dist = optimal_thresholds['score_distribution']
        if 'relevant' in dist and dist['relevant'].get('mean') is not None:
            print(f"  Relevant scores: mean={dist['relevant']['mean']:.3f}, std={dist['relevant']['std']:.3f}")
            print(f"                   min={dist['relevant']['min']:.3f}, max={dist['relevant']['max']:.3f}")
        if 'irrelevant' in dist and dist['irrelevant'].get('mean') is not None:
            print(f"  Irrelevant scores: mean={dist['irrelevant']['mean']:.3f}, std={dist['irrelevant']['std']:.3f}")
            print(f"                     min={dist['irrelevant']['min']:.3f}, max={dist['irrelevant']['max']:.3f}")

        print("\nOptimal Thresholds by Method:")
        for method, threshold in optimal_thresholds['methods'].items():
            print(f"  {method:30s}: {threshold:.3f}")

        print(f"\n  CONSENSUS THRESHOLD: {optimal_thresholds['consensus_threshold']:.3f}")

        consensus_metrics = optimal_thresholds['consensus_metrics']
        print(f"\nMetrics at Consensus Threshold:")
        print(f"  Precision: {consensus_metrics['precision']:.3f}")
        print(f"  Recall: {consensus_metrics['recall']:.3f}")
        print(f"  F1 Score: {consensus_metrics['f1']:.3f}")
        print(f"  Accuracy: {consensus_metrics['accuracy']:.3f}")

        # Compare with current threshold
        # Use macro-averaging for accurate comparison with actual system performance
        current_metrics = threshold_optimizer.calculate_metrics_at_threshold(
            self.settings.primary_similarity_threshold,
            use_macro_averaging=True
        )
        print(f"\nMetrics at Current Threshold ({self.settings.primary_similarity_threshold}):")
        print(f"  Precision: {current_metrics['precision']:.3f}")
        print(f"  Recall: {current_metrics['recall']:.3f}")
        print(f"  F1 Score: {current_metrics['f1']:.3f}")
        print(f"  Accuracy: {current_metrics['accuracy']:.3f}")

        # Improvement potential
        f1_improvement = consensus_metrics['f1'] - current_metrics['f1']
        if f1_improvement > 0:
            print(f"\n  💡 Switching to {optimal_thresholds['consensus_threshold']:.3f} would improve F1 by {f1_improvement:.3f}")

        # Save results
        output_dir = Path("evaluation_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"toolbench_eval_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "summary": summary,
                "results": results
            }, f, indent=2)

        print(f"\nResults saved to {output_file}")

        # Save threshold optimization analysis
        threshold_file = output_dir / f"threshold_analysis_{timestamp}.json"
        threshold_optimizer.save_analysis(str(threshold_file))
        print(f"Threshold analysis saved to {threshold_file}")

        return {
            "summary": summary,
            "results": results,
            "optimal_thresholds": optimal_thresholds
        }


async def main():
    """Main function to run ToolBench evaluation."""
    evaluator = ToolBenchEvaluator()

    # Test with one example first (from retrieval/G1) - without noise
    # print("Testing with one example (no noise in available_tools)...")
    # await evaluator.run_evaluation(
    #     test_file="one_example.json",
    #     data_dir="toolbench_data/data/retrieval/G1",
    #     num_cases=1,
    #     clear_collection=True,
    #     add_noise_tools=True,  # Add noise to vector store
    #     add_noise_to_available=0  # No noise in available_tools
    # )

    # Test with one example with noise in available_tools
    # print("\n\nTesting with one example (WITH noise in available_tools)...")
    # await evaluator.run_evaluation(
    #     test_file="one_example.json",
    #     data_dir="toolbench_data/data/retrieval/G1",
    #     num_cases=1,
    #     clear_collection=True,
    #     add_noise_tools=True,  # Add noise to vector store
    #     add_noise_to_available=10  # Add 10 noise tools to available_tools
    # )

    # Run full evaluation without noise in available_tools (baseline)
    # print("\n\nRunning evaluation WITHOUT noise in available_tools (baseline)...")
    # await evaluator.run_evaluation(
    #     test_file="G2_instruction.json",
    #     data_dir="toolbench_data/data/test_instruction",
    #     num_cases=20,
    #     clear_collection=True,
    #     add_noise_tools=True,  # Add noise to vector store
    #     add_noise_to_available=0  # No noise in available_tools
    # )
    #
    # Run full evaluation WITH noise in available_tools
    print("\n\nRunning evaluation WITH noise in available_tools...")
    await evaluator.run_evaluation(
        test_file="G2_instruction.json",
        data_dir="toolbench_data/data/test_instruction",
        num_cases=20,
        clear_collection=True,
        add_noise_tools=False,  # Add noise to vector store
        add_noise_to_available=0  # Add 100 noise tools to available_tools per test
    )


if __name__ == "__main__":
    asyncio.run(main())
