#!/usr/bin/env python3
"""Script to test LTR model performance and compare with other strategies."""

import asyncio
import json
import logging
from pathlib import Path
import random
from typing import List, Dict, Any
import time

from src.services.search_service import SearchService, SearchStrategy
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.bm25_ranker import BM25Ranker
from src.services.cross_encoder_reranker import CrossEncoderReranker
from src.services.ltr_service import LTRService
from src.evaluation.toolbench_evaluator import ToolBenchEvaluator
from src.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


async def test_ltr_on_queries():
    """Test LTR model on ToolBench test queries."""

    # Initialize services
    vector_store = VectorStoreService()
    embedding_service = EmbeddingService()
    bm25_ranker = BM25Ranker()
    cross_encoder = CrossEncoderReranker()

    # Initialize LTR service
    ltr_service = LTRService(
        model_path="./models/ltr_xgboost",
        bm25_ranker=bm25_ranker,
        cross_encoder=cross_encoder,
        auto_load=True
    )

    # Check if model is trained
    if not ltr_service.is_trained:
        logger.error("LTR model not found. Please run train_ltr.py first!")
        return

    # Create search service with LTR
    search_service = SearchService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_ranker=bm25_ranker,
        cross_encoder=cross_encoder,
        ltr_service=ltr_service
    )

    # Enable LTR in settings
    settings.enable_ltr = True

    # Load test cases from ToolBench (use a different file than training)
    evaluator = ToolBenchEvaluator()
    import json
    from slugify import slugify

    # Use G3 for testing (assuming G1 and G2 were used for training)
    test_file = "toolbench_data/data/test_instruction/G2_category.json"
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    # Select a few test cases for quick testing
    num_test_cases = 5
    test_cases = test_data[:num_test_cases]

    logger.info(f"Testing with {num_test_cases} test cases from {test_file}\n")

    # Test each strategy
    strategies_to_test = [
        SearchStrategy.SEMANTIC,
        SearchStrategy.HYBRID,
        SearchStrategy.HYBRID_CROSS_ENCODER,
        SearchStrategy.LTR,
        SearchStrategy.HYBRID_LTR
    ]

    results_by_strategy = {}

    for strategy in strategies_to_test:
        if strategy in [SearchStrategy.LTR, SearchStrategy.HYBRID_LTR] and not ltr_service.is_trained:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Testing strategy: {strategy.value}")
        logger.info(f"{'='*60}")

        strategy_results = []
        total_time = 0

        for idx, test_case in enumerate(test_cases):
            # Clear collection for clean test (skip first iteration)
            try:
                vector_store.client.delete_collection(vector_store.collection_name)
                await vector_store.initialize()
            except Exception as e:
                logger.warning(f"Could not clear collection: {e}")

            noise_tools_pool = await evaluator.load_random_toolbench_tools(
              target_total=300 * 3  # Load 3x the needed amount for variety
            )

            noise_tools = random.sample(noise_tools_pool, 300)

            await evaluator.index_tools(noise_tools, vector_store=vector_store, embedding_service=embedding_service)

            # Extract query and expected tools
            query = test_case.get("query", "")
            relevant_apis = test_case.get("relevant APIs", [])

            # Convert expected API references to tool names
            expected_tool_names = set()
            for api_ref in relevant_apis:
                if len(api_ref) >= 2:
                    tool_name = api_ref[0]
                    api_name = api_ref[1]
                    raw_name = f"{tool_name}_{api_name}"
                    expected_name = slugify(raw_name, separator="_")
                    expected_tool_names.add(expected_name)

            # Index tools for this test case
            available_tools = await evaluator.index_tools_from_case(
                test_case=test_case,
                vector_store=vector_store,
                embedding_service=embedding_service
            )

            if not available_tools:
                logger.warning(f"No tools indexed for test case {idx}, skipping")
                continue

            available_tools.extend(noise_tools)

            logger.info(f"\nTest {idx+1}: Query: {query[:80]}...")
            logger.info(f"  Expected tools: {list(expected_tool_names)[:3]}...")
            logger.info(f"  Available tools: {len(available_tools)}")

            # Time the search
            start_time = time.time()
            results = await search_service.search(
                query=query,
                available_tools=available_tools,
                strategy=strategy,
                limit=10,
                score_threshold=0.0  # Don't filter by score for testing
            )
            search_time = time.time() - start_time
            total_time += search_time

            # Extract tool names from results
            found_tools = []
            for r in results:  # Check top 5
                if 'tool_name' in r:
                    found_tools.append(r['tool_name'])
                elif 'name' in r:
                    found_tools.append(r['name'])

            # Check if expected tools were found
            found_expected = [t for t in expected_tool_names if t in found_tools]
            success = len(found_expected) > 0

            metrics = evaluator.evaluate_results(query=query, expected_apis=relevant_apis, recommended_tools=results)
            metrics.update({
                "query": query,
                "expected": list(expected_tool_names),
                "found": found_tools,
                "found_expected_num": len(found_expected),
                "success": success,
                "time_ms": search_time * 1000
            })
            strategy_results.append(metrics)

            # Log result
            status = "✓" if success else "✗"
            logger.info(f"  {status} Expected: {expected_tool_names}")
            logger.info(f"  {status} Found Expected: {found_expected}")
            logger.info(f"  {status} Found: {found_tools}")
            logger.info(f"  Time: {search_time*1000:.1f}ms")

            if strategy in [SearchStrategy.LTR, SearchStrategy.HYBRID_LTR] and results:
                # Show LTR score for top result
                top_result = results[0]
                if 'ltr_score' in top_result:
                    logger.info(f"  LTR Score: {top_result['ltr_score']:.3f}")

        # Summary for strategy
        if strategy_results:
            success_rate = sum(1 for r in strategy_results if r['success']) / len(strategy_results)
            avg_time = total_time / len(strategy_results) * 1000
        else:
            success_rate = 0
            avg_time = 0

        logger.info(f"\n{strategy.value} Summary:")
        logger.info(f"  Success Rate: {success_rate*100:.1f}%")
        logger.info(f"  Avg Time: {avg_time:.1f}ms")

        results_by_strategy[strategy.value] = {
            "success_rate": success_rate,
            "avg_time_ms": avg_time,
            "results": strategy_results
        }

    # Compare strategies
    logger.info(f"\n{'='*60}")
    logger.info("STRATEGY COMPARISON")
    logger.info(f"{'='*60}")

    logger.info(f"{'Strategy':<25} {'Success Rate':<15} {'Avg Time (ms)':<15}")
    logger.info("-" * 55)

    for strategy, metrics in results_by_strategy.items():
        logger.info(
            f"{strategy:<25} "
            f"{metrics['success_rate']*100:>6.1f}%        "
            f"{metrics['avg_time_ms']:>8.1f}"
        )

    # Show LTR improvements if available
    if "LTR" in results_by_strategy and "SEMANTIC" in results_by_strategy:
        ltr_improvement = (
            results_by_strategy["LTR"]["success_rate"] -
            results_by_strategy["SEMANTIC"]["success_rate"]
        ) * 100
        logger.info(f"\nLTR Improvement over Semantic: {ltr_improvement:+.1f}%")

    if "HYBRID_LTR" in results_by_strategy and "HYBRID_CROSS_ENCODER" in results_by_strategy:
        hybrid_ltr_improvement = (
            results_by_strategy["HYBRID_LTR"]["success_rate"] -
            results_by_strategy["HYBRID_CROSS_ENCODER"]["success_rate"]
        ) * 100
        logger.info(f"Hybrid-LTR Improvement over Hybrid-CE: {hybrid_ltr_improvement:+.1f}%")


async def run_full_evaluation():
    """Run full ToolBench evaluation with LTR enabled."""

    logger.info("Running full ToolBench evaluation with LTR...")

    # Check if LTR model exists
    model_path = Path("./models/ltr_xgboost.json")
    if not model_path.exists():
        logger.error("LTR model not found. Please run train_ltr.py first!")
        return

    # Enable LTR in settings
    settings.enable_ltr = True

    # Initialize evaluator
    evaluator = ToolBenchEvaluator()

    # Initialize search service with LTR
    vector_store = VectorStoreService()
    embedding_service = EmbeddingService()
    bm25_ranker = BM25Ranker()
    cross_encoder = CrossEncoderReranker()
    ltr_service = LTRService(
        model_path="./models/ltr_xgboost",
        bm25_ranker=bm25_ranker,
        cross_encoder=cross_encoder,
        auto_load=True
    )

    search_service = SearchService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_ranker=bm25_ranker,
        cross_encoder=cross_encoder,
        ltr_service=ltr_service
    )

    # Set the search service in evaluator
    evaluator.search_service = search_service

    # Run evaluation with all strategies
    results = await evaluator.run_evaluation(
        test_queries_limit=50,  # Test on 50 queries
        noise_tools_count=100  # Add 100 noise tools to test robustness
    )

    # Display results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS WITH NOISE")
    logger.info("="*60)

    for strategy_result in results:
        strategy = strategy_result['strategy']
        metrics = strategy_result['metrics']

        logger.info(f"\n{strategy}:")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall: {metrics['recall']:.3f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.3f}")
        logger.info(f"  Expected Tool Recall: {metrics.get('expected_tool_recall', 0):.3f}")

    # Save results
    output_path = Path("results/ltr_evaluation_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")


async def test_specific_improvements():
    """Test specific improvements that LTR should handle better."""

    logger.info("\nTesting specific LTR improvements...")

    # Initialize services
    vector_store = VectorStoreService()
    embedding_service = EmbeddingService()
    search_service = SearchService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        ltr_service=LTRService(auto_load=True)
    )

    # Test exact name matching
    logger.info("\n1. Testing exact name matching...")
    query = "search_files"  # Exact tool name

    semantic_results = await search_service.search(
        query=query,
        strategy=SearchStrategy.SEMANTIC,
        limit=5
    )

    ltr_results = await search_service.search(
        query=query,
        strategy=SearchStrategy.LTR,
        limit=5
    )

    logger.info(f"Query: '{query}'")
    logger.info(f"Semantic top result: {semantic_results[0].get('tool_name', 'N/A') if semantic_results else 'None'}")
    logger.info(f"LTR top result: {ltr_results[0].get('tool_name', 'N/A') if ltr_results else 'None'}")

    # Test parameter matching
    logger.info("\n2. Testing parameter name matching...")
    query = "function with pattern and recursive parameters"

    semantic_results = await search_service.search(
        query=query,
        strategy=SearchStrategy.SEMANTIC,
        limit=5
    )

    ltr_results = await search_service.search(
        query=query,
        strategy=SearchStrategy.LTR,
        limit=5
    )

    logger.info(f"Query: '{query}'")
    logger.info("Semantic top 3:")
    for i, r in enumerate(semantic_results[:3], 1):
        logger.info(f"  {i}. {r.get('tool_name', 'N/A')}")

    logger.info("LTR top 3:")
    for i, r in enumerate(ltr_results[:3], 1):
        logger.info(f"  {i}. {r.get('tool_name', 'N/A')}")


async def main():
    """Main entry point for testing."""

    # Parse command line arguments
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "quick":
            await test_ltr_on_queries()
        elif mode == "full":
            await run_full_evaluation()
        elif mode == "specific":
            await test_specific_improvements()
        else:
            logger.error(f"Unknown mode: {mode}")
            logger.info("Usage: python test_ltr.py [quick|full|specific]")
    else:
        # Default: run quick test
        await test_ltr_on_queries()


if __name__ == "__main__":
    asyncio.run(main())
