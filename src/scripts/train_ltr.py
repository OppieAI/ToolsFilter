#!/usr/bin/env python3
"""Script to train LTR model from evaluation data."""
from slugify import slugify
import asyncio
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.evaluation.toolbench_evaluator import ToolBenchEvaluator
from src.core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()


async def generate_training_data_from_toolbench(
    instruction_files: List[str] = None,
    num_queries_per_file: int = 100,
    save_path: str = "data/ltr_training_data.json"
) -> List[Dict[str, Any]]:
    """
    Generate training data from ToolBench test instruction files.

    NO SEARCH NEEDED - we already have labeled data:
    - Query
    - All available tools (api_list)
    - Ground truth relevant tools (relevant APIs)

    Args:
        instruction_files: List of paths to instruction files. If None, uses all G*.json files
        num_queries_per_file: Number of queries to use per file for training
        save_path: Path to save training data

    Returns:
        List of evaluation results for training
    """
    # Check if we already have saved training data
    if save_path and Path(save_path).exists():
        logger.info(f"Loading existing training data from {save_path}")
        with open(save_path, 'r') as f:
            return json.load(f)

    # Default to all G*.json files if none specified
    if instruction_files is None:
        instruction_dir = Path("toolbench_data/data/test_instruction")
        instruction_files = sorted(instruction_dir.glob("G*.json"))
        instruction_files = [str(f) for f in instruction_files]
        logger.info(f"Found {len(instruction_files)} instruction files: {[Path(f).name for f in instruction_files]}")

    # Initialize ToolBenchEvaluator for API conversion
    evaluator = ToolBenchEvaluator()

    noise_tools_pool = await evaluator.load_random_toolbench_tools(
      target_total=500 * 3  # Load 3x the needed amount for variety
    )

    # Process test cases from all files
    all_training_data = []

    for file_path in instruction_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Loading ToolBench data from {Path(file_path).name}")

        # Load instruction data
        with open(file_path, 'r') as f:
            instruction_data = json.load(f)

        logger.info(f"Loaded {len(instruction_data)} test cases from {Path(file_path).name}")

        # Process test cases - NO SEARCH, just prepare labeled data
        file_training_data = []

        for idx, test_case in enumerate(instruction_data[:num_queries_per_file]):
            if (idx + 1) % 10 == 0:  # Log progress every 10 queries
                logger.info(f"Processing {idx+1}/{min(num_queries_per_file, len(instruction_data))} from {Path(file_path).name}")

            # Extract query
            query = test_case.get("query", "")
            query_id = test_case.get("query_id", f"{Path(file_path).stem}_query_{idx}")

            # Get expected APIs (ground truth labels)
            relevant_apis = test_case.get("relevant APIs", [])
            expected_tool_names = set()
            for api_ref in relevant_apis:
                if len(api_ref) >= 2:
                    tool_name = api_ref[0]
                    api_name = api_ref[1]
                    raw_name = f"{tool_name}_{api_name}"
                    # Use slugify to normalize the same way as when creating tools
                    expected_name = slugify(raw_name, separator="_")
                    expected_tool_names.add(expected_name)

            # Convert ALL APIs to OpenAI tool format
            api_list = test_case.get("api_list", [])
            all_tools = []
            for api in api_list:
                try:
                    tool = evaluator.convert_api_to_openai_format(api)
                    tool_dict = tool.model_dump()

                    # Extract tool name for matching
                    if tool_dict.get("type") == "function":
                        if "function" in tool_dict:
                            tool_name = tool_dict["function"].get("name", "")
                        else:
                            tool_name = tool_dict.get("name", "")
                    else:
                        tool_name = api.get("api_name", "")

                    # Add relevance label (1 if in expected, 0 otherwise)
                    tool_dict['relevance_label'] = 1.0 if tool_name in expected_tool_names else 0.0
                    tool_dict['tool_name'] = tool_name

                    all_tools.append(tool_dict)
                except Exception as e:
                    logger.warning(f"Error converting API: {e}")
                    continue

            noise_tools = random.sample(noise_tools_pool, 300)
            for tool in noise_tools:
              try:
                tool_dict = tool.model_dump()
                tool_name = ""
                # Extract tool name for matching
                if tool_dict.get("type") == "function":
                  if "function" in tool_dict:
                    tool_name = tool_dict["function"].get("name", "")
                  else:
                    tool_name = tool_dict.get("name", "")

                # Add relevance label (1 if in expected, 0 otherwise)
                tool_dict['relevance_label'] = 1.0 if tool_name in expected_tool_names else 0.0
                tool_dict['tool_name'] = tool_name

                all_tools.append(tool_dict)
              except Exception as e:
                logger.warning(f"Error converting Noise tools: {e}")
                continue

            if not all_tools:
                logger.warning(f"No tools converted for query {query_id}, skipping")
                continue

            # Store training sample with labeled data
            training_sample = {
                'query_id': query_id,
                'query': query,
                'available_tools': all_tools,  # All tools with relevance labels
                'expected_tools': list(expected_tool_names),  # Convert set to list for JSON serialization
                'num_tools': len(all_tools),
                'num_relevant': len(expected_tool_names)
            }
            file_training_data.append(training_sample)

        # Add this file's data to overall training data
        all_training_data.extend(file_training_data)
        logger.info(f"Processed {len(file_training_data)} queries from {Path(file_path).name}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Total training samples collected: {len(all_training_data)}")

    # Save training data
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_training_data, f, indent=2)
        logger.info(f"\nSaved {len(all_training_data)} training samples to {save_path}")

    return all_training_data


async def train_ltr_model():
    """Train LTR model from labeled ToolBench data."""

    # Generate or load training data from all available files
    training_data = await generate_training_data_from_toolbench(
        instruction_files=None,  # Use all G*.json files
        num_queries_per_file=50,  # Use 50 queries from each file
        save_path="data/ltr_training_data.json"
    )

    if len(training_data) < 50:
        logger.warning(f"Only {len(training_data)} training samples. Consider generating more data.")



    # Initialize components for feature extraction
    from src.services.ltr_feature_extractor import LTRFeatureExtractor
    from src.services.ltr_trainer import LTRTrainer

    feature_extractor = LTRFeatureExtractor()
    trainer = LTRTrainer(
        model_type="xgboost",
        objective="rank:pairwise"
    )

    # Convert training data to LTR format
    logger.info("Preparing training data for LTR...")
    all_features = []
    all_relevance = []
    query_groups = []

    for sample in training_data:
        query = sample['query']
        available_tools = sample['available_tools']


        # Extract features for each tool
        query_features = []
        query_relevance = []

        for tool in available_tools:
            # Extract features
            features = feature_extractor.extract_features(query, tool)
            feature_values = list(features.values())
            query_features.append(feature_values)

            # Get relevance label (already set in the tool dict)
            relevance = tool.get('relevance_label', 0.0)
            query_relevance.append(relevance)

        if query_features:
            all_features.extend(query_features)
            all_relevance.extend(query_relevance)
            query_groups.append(len(query_features))

    # Convert to numpy arrays
    import numpy as np
    X = np.array(all_features)
    y = np.array(all_relevance)

    logger.info(f"Training data shape: {X.shape}")
    logger.info(f"Number of queries: {len(query_groups)}")
    logger.info(f"Positive samples: {np.sum(y)}/{len(y)} ({np.sum(y)/len(y)*100:.1f}%)")

    # Train the model
    logger.info("Starting LTR model training...")
    metrics = trainer.train(
        features=X,
        relevance_scores=y,
        query_groups=query_groups,
        feature_names=feature_extractor.get_feature_names(),
        validation_split=0.2,
        early_stopping_rounds=10
    )

    # Save the model
    model_path = "./models/ltr_xgboost"
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Display training metrics
    logger.info("\nTraining completed!")
    logger.info(f"Training metrics: {json.dumps(metrics.get('train_metrics', {}), indent=2)}")
    logger.info(f"Validation metrics: {json.dumps(metrics.get('validation_metrics', {}), indent=2)}")

    # Display feature importance
    if metrics.get('feature_importance'):
        logger.info("\nTop 15 most important features:")
        importance_sorted = sorted(
            [(k.replace('_gain', ''), v) for k, v in metrics['feature_importance'].items() if k.endswith('_gain')],
            key=lambda x: x[1],
            reverse=True
        )[:15]
        for i, (feature, importance) in enumerate(importance_sorted, 1):
            logger.info(f"{i:2}. {feature:30} : {importance:.2f}")

    # Cross-validation if we have enough data
    if len(training_data) >= 50:
        logger.info("\nRunning 5-fold cross-validation...")
        cv_metrics = trainer.cross_validate(X, y, query_groups, cv_folds=5)
        logger.info(f"CV NDCG@10: {cv_metrics['mean_ndcg@10']:.4f} ± {cv_metrics['std_ndcg@10']:.4f}")

    return metrics


if __name__ == "__main__":
    asyncio.run(train_ltr_model())
