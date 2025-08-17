"""Run evaluation on the PTR Tool Filter API."""

import asyncio
import httpx
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.evaluation.metrics import ToolFilterEvaluator, EVALUATION_TEST_CASES
from src.services.tool_loader import get_sample_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIEvaluator:
    """Evaluator that tests the API with various scenarios."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.evaluator = ToolFilterEvaluator()
        self.available_tools = self._prepare_tools()
    
    def _prepare_tools(self) -> List[Dict[str, Any]]:
        """Prepare available tools in API format."""
        sample_tools = get_sample_tools()
        return [tool.dict() for tool in sample_tools]
    
    async def run_single_evaluation(
        self,
        test_case: Dict[str, Any],
        max_tools: int = 5
    ) -> Dict[str, Any]:
        """Run a single evaluation test case."""
        async with httpx.AsyncClient() as client:
            request_data = {
                "messages": test_case["messages"],
                "available_tools": self.available_tools,
                "max_tools": max_tools,
                "include_reasoning": True
            }
            
            try:
                response = await client.post(
                    f"{self.api_url}/api/v1/tools/filter",
                    json=request_data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract predicted tools and scores
                    predicted_tools = [
                        tool["tool_name"] 
                        for tool in result["recommended_tools"]
                    ]
                    similarity_scores = [
                        tool["confidence"] 
                        for tool in result["recommended_tools"]
                    ]
                    processing_time = result["metadata"]["processing_time_ms"]
                    
                    # Evaluate
                    eval_result = self.evaluator.evaluate_tool_recommendations(
                        predicted_tools=predicted_tools,
                        ground_truth_tools=test_case["expected_tools"],
                        similarity_scores=similarity_scores,
                        processing_time_ms=processing_time,
                        k=max_tools,
                        metadata={
                            "test_case": test_case["name"],
                            "description": test_case["description"]
                        }
                    )
                    
                    return {
                        "test_case": test_case["name"],
                        "status": "success",
                        "predicted_tools": predicted_tools,
                        "expected_tools": test_case["expected_tools"],
                        "metrics": {
                            "precision": eval_result.precision_at_k,
                            "recall": eval_result.recall_at_k,
                            "f1_score": eval_result.f1_score,
                            "mrr": eval_result.mean_reciprocal_rank,
                            "ndcg": eval_result.ndcg_score,
                            "avg_similarity": eval_result.average_similarity,
                            "processing_time_ms": eval_result.processing_time_ms
                        }
                    }
                else:
                    return {
                        "test_case": test_case["name"],
                        "status": "error",
                        "error": f"API returned {response.status_code}: {response.text}"
                    }
                    
            except Exception as e:
                return {
                    "test_case": test_case["name"],
                    "status": "error",
                    "error": str(e)
                }
    
    async def run_full_evaluation(self):
        """Run evaluation on all test cases."""
        logger.info(f"Starting evaluation with {len(EVALUATION_TEST_CASES)} test cases")
        
        results = []
        for test_case in EVALUATION_TEST_CASES:
            logger.info(f"Running test case: {test_case['name']}")
            result = await self.run_single_evaluation(test_case)
            results.append(result)
            
            # Print individual result
            if result["status"] == "success":
                logger.info(f"  ✓ Precision: {result['metrics']['precision']:.2f}")
                logger.info(f"  ✓ Recall: {result['metrics']['recall']:.2f}")
                logger.info(f"  ✓ F1 Score: {result['metrics']['f1_score']:.2f}")
                logger.info(f"  ✓ NDCG: {result['metrics']['ndcg']:.2f}")
                logger.info(f"  ✓ Processing time: {result['metrics']['processing_time_ms']:.2f}ms")
            else:
                logger.error(f"  ✗ Error: {result['error']}")
        
        # Get aggregate metrics
        aggregate_metrics = self.evaluator.get_aggregate_metrics()
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total test cases: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")
        
        if aggregate_metrics:
            print("\nAggregate Metrics:")
            print(f"  Average Precision@5: {aggregate_metrics['avg_precision']:.3f}")
            print(f"  Average Recall@5: {aggregate_metrics['avg_recall']:.3f}")
            print(f"  Average F1 Score: {aggregate_metrics['avg_f1']:.3f}")
            print(f"  Average MRR: {aggregate_metrics['avg_mrr']:.3f}")
            print(f"  Average NDCG: {aggregate_metrics['avg_ndcg']:.3f}")
            print(f"  Average Similarity: {aggregate_metrics['avg_similarity']:.3f}")
            print(f"  Average Processing Time: {aggregate_metrics['avg_processing_time_ms']:.2f}ms")
            print(f"  P95 Processing Time: {aggregate_metrics['p95_processing_time_ms']:.2f}ms")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Export detailed results
        detailed_output = f"{output_dir}/evaluation_results_{timestamp}.json"
        with open(detailed_output, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "test_results": results,
                "aggregate_metrics": aggregate_metrics
            }, f, indent=2)
        
        # Export metrics history
        metrics_output = f"{output_dir}/metrics_history_{timestamp}.json"
        self.evaluator.export_metrics(metrics_output)
        
        print(f"\nResults exported to:")
        print(f"  - {detailed_output}")
        print(f"  - {metrics_output}")
        
        return results, aggregate_metrics


async def main():
    """Main evaluation function."""
    # Check if API is healthy
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/health")
            if response.status_code != 200:
                logger.error("API is not healthy. Please ensure the API is running.")
                return
        except Exception as e:
            logger.error(f"Cannot connect to API: {e}")
            return
    
    # Run evaluation
    evaluator = APIEvaluator()
    await evaluator.run_full_evaluation()


if __name__ == "__main__":
    asyncio.run(main())