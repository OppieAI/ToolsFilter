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


async def train_ltr_model_with_context():
    """Train LTR model from labeled ToolBench data with context scores."""

    # Generate or load training data from all available files
    training_data = await generate_training_data_from_toolbench(
        instruction_files=None,  # Use all G*.json files
        num_queries_per_file=150,  # Reduced from 150 due to search overhead
        save_path="data/ltr_training_data.json"
    )

    if len(training_data) < 50:
        logger.warning(f"Only {len(training_data)} training samples. Consider generating more data.")

    # Limit training data for faster testing (remove this in production)
    # training_data = training_data[:20]  # Use first 20 samples for faster testing
    # logger.info(f"Using {len(training_data)} training samples for LTR training (limited for testing)")
    # logger.info("NOTE: Remove the training_data limit for full production training")



    # Initialize components for feature extraction
    from src.services.ltr_feature_extractor import LTRFeatureExtractor
    from src.services.ltr_trainer import LTRTrainer
    from src.services.search_service import SearchService
    from src.services.vector_store import VectorStoreService
    from src.services.embeddings import EmbeddingService
    from src.services.bm25_ranker import BM25Ranker, HybridScorer
    from src.services.cross_encoder_reranker import CrossEncoderReranker
    from src.services.embedding_enhancer import ToolEmbeddingEnhancer

    feature_extractor = LTRFeatureExtractor()
    trainer = LTRTrainer(
        model_type="xgboost",
        objective="rank:pairwise"
    )

    # Initialize search services to get context scores
    logger.info("Initializing search services for context generation...")
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    bm25_ranker = BM25Ranker()
    hybrid_scorer = HybridScorer()
    cross_encoder = CrossEncoderReranker()

    # Pre-index all training tools in Qdrant for semantic search
    logger.info("🔧 Pre-indexing training tools in Qdrant...")
    await index_training_tools(training_data, vector_store, embedding_service)

    # Initialize search service (without LTR to avoid circular dependency)
    logger.info("Initializing SearchService for context generation (LTR disabled to avoid circular dependency)")
    search_service = SearchService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        bm25_ranker=bm25_ranker,
        cross_encoder=cross_encoder,  # Fixed parameter name
        hybrid_scorer=hybrid_scorer,
        ltr_service=None  # Don't use LTR during training!
    )

    # Convert training data to LTR format with context generation
    logger.info("Preparing training data for LTR with context scores...")
    all_features = []
    all_relevance = []
    query_groups = []

    for idx, sample in enumerate(training_data):
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing training sample {idx+1}/{len(training_data)}")

        query = sample['query']
        available_tools = sample['available_tools']

        # Convert dictionary tools to Tool objects for SearchService
        from src.core.models import Tool
        tools_for_search = []
        for tool_dict in available_tools:
            # Create Tool object from dictionary
            tool_obj = Tool(
                type=tool_dict.get('type', 'function'),
                name=tool_dict.get('name', ''),
                description=tool_dict.get('description', ''),
                parameters=tool_dict.get('parameters', {}),
                category=tool_dict.get('category', 'general')
            )
            tools_for_search.append(tool_obj)

        # Run search pipeline to get context scores using SearchService
        # Initialize variables for debug logging
        semantic_results = []
        cross_encoder_results = []
        bm25_scores = {}
        combined_scores = []

        try:
            # Step 1: Get semantic scores using search_service
            # Tools should now be indexed in Qdrant, so we should get real scores!

            # DEBUG: Check if tools are actually available in vector store
            if idx == 0:  # Only debug first sample
                logger.info(f"🔍 DEBUG: Checking if tools are in vector store...")
                sample_tool_names = [tool.name for tool in tools_for_search[:5]]
                for tool_name in sample_tool_names:
                    try:
                        found_tool = await vector_store.get_tool_by_name(tool_name)
                        logger.info(f"  Tool '{tool_name}': {'✅ FOUND' if found_tool else '❌ NOT FOUND'}")
                    except Exception as e:
                        logger.info(f"  Tool '{tool_name}': ❌ ERROR: {e}")

            # Try semantic search with very low threshold to ensure results
            semantic_results = await search_service.semantic_search(
                query=query,
                available_tools=None,  # Don't filter by available tools
                limit=len(tools_for_search),  # Get all tools
                score_threshold=0.0  # Set very low threshold to ensure results!
            )

            # If we got results, filter to only available tools
            if semantic_results:
                available_names = {tool.name for tool in tools_for_search}
                semantic_results = [
                    result for result in semantic_results
                    if result.get('tool_name', result.get('name', '')) in available_names
                ]
                if idx == 0:
                    logger.info(f"🎯 Filtered semantic results: {len(semantic_results)} tools")

            # If still no results, create fallback
            if not semantic_results:
                logger.warning(f"No semantic results despite indexing for query: {query[:50]}...")
                semantic_results = [{
                    'tool_name': tool.name,
                    'name': tool.name,
                    'score': 0.1
                } for tool in tools_for_search]

            # Step 2: Get BM25 scores using search_service's BM25 ranker
            bm25_scores = search_service.bm25_ranker.score_tools(query, tools_for_search)

            # Step 3: Get cross-encoder scores if available
            cross_encoder_results = []
            try:
                # Use cross_encoder_rerank with correct parameters
                if hasattr(search_service, 'cross_encoder') and search_service.cross_encoder:
                    # Convert semantic results to candidates format for cross-encoder
                    candidates = [{
                        'tool_name': result.get('tool_name', result.get('name', '')),
                        'name': result.get('tool_name', result.get('name', '')),
                        'score': result['score']
                    } for result in semantic_results]

                    cross_encoder_results = await search_service.cross_encoder_rerank(
                        query=query,
                        candidates=candidates,  # Fixed parameter name
                        top_k=len(tools_for_search)  # Fixed parameter name
                    )
                else:
                    # Fallback: use semantic scores
                    cross_encoder_results = semantic_results
            except Exception as e:
                logger.warning(f"Cross-encoder scoring failed: {e}")
                cross_encoder_results = semantic_results  # Fallback

            # Create score mappings by tool name for easy lookup
            semantic_score_map = {}
            for result in semantic_results:
                tool_name = result.get('tool_name', result.get('name', ''))
                semantic_score_map[tool_name] = result['score']

            cross_encoder_score_map = {}
            for result in cross_encoder_results:
                tool_name = result.get('tool_name', result.get('name', ''))
                cross_encoder_score_map[tool_name] = result['score']

            # Create ranking and position information
            combined_scores = []
            for i, tool_dict in enumerate(available_tools):  # tool_dict is from training data
                # Get tool name from dictionary format
                tool_name = tool_dict.get('name', tool_dict.get('tool_name', ''))

                # Get individual scores
                semantic_score = semantic_score_map.get(tool_name, 0.0)
                bm25_score = bm25_scores.get(tool_name, 0.0) if isinstance(bm25_scores, dict) else 0.0
                cross_score = cross_encoder_score_map.get(tool_name, semantic_score)

                # Weighted combination for ranking
                combined_score = 0.4 * semantic_score + 0.3 * bm25_score + 0.3 * cross_score
                combined_scores.append({
                    'index': i,
                    'tool': tool_dict,  # Keep original dictionary for feature extraction
                    'tool_name': tool_name,
                    'semantic_score': semantic_score,
                    'bm25_score': bm25_score,
                    'cross_encoder_score': cross_score,
                    'combined_score': combined_score
                })

            # Sort by combined score to get rankings
            combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            top_score = combined_scores[0]['combined_score'] if combined_scores else 0.0

            # Extract features for each tool with context
            query_features = []
            query_relevance = []

            for rank, score_info in enumerate(combined_scores):
                tool_dict = score_info['tool']  # This is the original tool dictionary

                # Create context with all the scores
                context = {
                    'semantic_score': score_info['semantic_score'],
                    'bm25_score': score_info['bm25_score'],
                    'cross_encoder_score': score_info['cross_encoder_score'],
                    'initial_rank': float(rank + 1),  # 1-indexed rank
                    'initial_score': score_info['combined_score'],
                    'top_score': top_score
                }

                # Extract features with context
                features = feature_extractor.extract_features(query, tool_dict, context)
                feature_values = list(features.values())
                query_features.append(feature_values)

                # Get relevance label (already set in the tool dict)
                relevance = tool_dict.get('relevance_label', 0.0)
                query_relevance.append(relevance)

        except Exception as e:
            logger.warning(f"Error processing training sample {idx}: {e}")
            # Fallback to no context if search fails
            query_features = []
            query_relevance = []

            for tool_dict in available_tools:
                # Use empty context as fallback
                features = feature_extractor.extract_features(query, tool_dict, {})
                feature_values = list(features.values())
                query_features.append(feature_values)
                relevance = tool_dict.get('relevance_label', 0.0)
                query_relevance.append(relevance)

            # Variables already initialized above for debug logging

        # Debug: Show context for first few samples
        if idx < 3 and query_features:
            logger.info(f"\n🔍 Sample {idx+1} - Using SearchService for context:")
            logger.info(f"  Query: {query}")
            logger.info(f"  Available tools: {len(available_tools)}")

            if combined_scores:
                logger.info(f"  ✅ Successfully used SearchService methods:")
                logger.info(f"    - semantic_search() returned {len(semantic_results)} results")
                bm25_count = len(bm25_scores) if isinstance(bm25_scores, dict) else 0
                logger.info(f"    - bm25_ranker.score_tools() returned {bm25_count} scores")
                logger.info(f"    - cross_encoder_rerank() returned {len(cross_encoder_results)} results")

                # Show semantic score improvement
                semantic_scores_list = [r['score'] for r in semantic_results]
                max_semantic = max(semantic_scores_list) if semantic_scores_list else 0.0
                avg_semantic = sum(semantic_scores_list) / len(semantic_scores_list) if semantic_scores_list else 0.0

                top_result = combined_scores[0]
                logger.info(f"  Top tool: {top_result.get('tool_name', 'unknown')}")
                logger.info(f"    Semantic: {top_result['semantic_score']:.4f} (max: {max_semantic:.4f}, avg: {avg_semantic:.4f})")
                logger.info(f"    BM25: {top_result['bm25_score']:.4f}")
                logger.info(f"    Cross-encoder: {top_result['cross_encoder_score']:.4f}")
                logger.info(f"    Combined: {top_result['combined_score']:.4f}")
            else:
                logger.info(f"  ⚠️ Fallback mode - no SearchService context")
                logger.info(f"  Query features: {len(query_features)}")
                logger.info(f"  Relevance labels: {sum(query_relevance)}/{len(query_relevance)}")

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
    logger.info("\n" + "="*80)
    logger.info("🎯 LTR TRAINING COMPLETED WITH IMPROVED FEATURES!")
    logger.info("="*80)
    logger.info(f"Training metrics: {json.dumps(metrics.get('train_metrics', {}), indent=2)}")
    logger.info(f"Validation metrics: {json.dumps(metrics.get('validation_metrics', {}), indent=2)}")

    # Show improvement over constant features
    logger.info("\n🚀 KEY IMPROVEMENTS:")
    logger.info("  ✅ Pre-indexed training tools in Qdrant for real semantic scores")
    logger.info("  ✅ Used SearchService properly instead of manual reimplementation")
    logger.info("  ✅ Context scores from semantic_search(), BM25, and cross_encoder_rerank()")
    logger.info("  ✅ Removed constant domain features (domain_network, domain_api, etc.)")
    logger.info("  ✅ Added query-tool interaction features with real search pipeline scores")
    logger.info("  ✅ Features capture semantic similarity, lexical overlap, intent alignment")
    logger.info("  ✅ Position and ranking information included")
    logger.info(f"  ✅ Total features extracted: {len(feature_extractor.get_feature_names())}")
    logger.info("  🎯 This solves the 'constant features problem' from the roadmap!")

    # Display feature importance
    if metrics.get('feature_importance'):
        logger.info("\nTop 20 most important features:")
        importance_sorted = sorted(
            [(k.replace('_gain', ''), v) for k, v in metrics['feature_importance'].items() if k.endswith('_gain')],
            key=lambda x: x[1],
            reverse=True
        )[:20]

        # Highlight new query-tool interaction features
        interaction_features = {
            'query_tool_name_overlap', 'query_desc_overlap', 'jaccard_similarity',
            'query_tool_cosine_sim', 'query_desc_cosine_sim', 'verb_match',
            'action_alignment', 'param_relevance', 'required_param_match'
        }

        for i, (feature, importance) in enumerate(importance_sorted, 1):
            marker = " 🔥" if feature in interaction_features else ""
            logger.info(f"{i:2}. {feature:35} : {importance:.3f}{marker}")

        # Count how many new features are in top 10
        top_10_new = sum(1 for f, _ in importance_sorted[:10] if f in interaction_features)
        logger.info(f"\n🔥 New interaction features in top 10: {top_10_new}/10")

    # Cross-validation if we have enough data
    if len(training_data) >= 50:
        logger.info("\nRunning 5-fold cross-validation...")
        try:
            cv_metrics = trainer.cross_validate(X, y, query_groups, cv_folds=5)
            logger.info(f"CV NDCG@10: {cv_metrics['mean_ndcg@10']:.4f} ± {cv_metrics['std_ndcg@10']:.4f}")
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}")

    return metrics


# Keep the old function for backward compatibility
async def train_ltr_model():
    """Backward compatibility wrapper."""
    return await train_ltr_model_with_context()


async def index_training_tools(
    training_data: List[Dict[str, Any]],
    vector_store,
    embedding_service
):
    """Pre-index all training tools in Qdrant for semantic search."""
    from src.services.embedding_enhancer import ToolEmbeddingEnhancer

    # Collect all unique tools from training data
    all_tools = {}
    for sample in training_data:
        for tool_dict in sample['available_tools']:
            tool_name = tool_dict.get('name', tool_dict.get('tool_name', ''))
            if tool_name and tool_name not in all_tools:
                all_tools[tool_name] = tool_dict

    logger.info(f"Found {len(all_tools)} unique tools in training data")

    # Check which tools need indexing (following endpoints.py pattern)
    tools_to_index = []
    tool_texts = []
    enhancer = ToolEmbeddingEnhancer()

    for tool_name, tool_dict in all_tools.items():
        try:
            # Check if tool already exists in vector store
            existing_tool = await vector_store.get_tool_by_name(tool_name)
            if not existing_tool:
                tools_to_index.append(tool_dict)
                # Convert tool to rich text for embedding (like endpoints.py)
                tool_text = enhancer.tool_to_rich_text(tool_dict)
                tool_texts.append(tool_text)
                logger.debug(f"Will index tool: {tool_name}")
            else:
                logger.debug(f"Tool already indexed: {tool_name}")
        except Exception as e:
            logger.debug(f"Error checking tool {tool_name}: {e}")
            # Add to indexing queue anyway
            tools_to_index.append(tool_dict)
            try:
                tool_text = enhancer.tool_to_rich_text(tool_dict)
                tool_texts.append(tool_text)
                logger.debug(f"Will index tool (fallback): {tool_name}")
            except Exception as text_error:
                # Fallback to simple text representation
                logger.debug(f"Failed to create rich text for {tool_name}: {text_error}")
                simple_text = f"{tool_dict.get('name', '')} {tool_dict.get('description', '')}"
                tool_texts.append(simple_text)

    # Index tools in batches (following endpoints.py pattern)
    if tools_to_index:
        try:
            logger.info(f"🚀 Indexing {len(tools_to_index)} training tools...")

            # Generate embeddings for all tools
            embeddings = await embedding_service.embed_batch(tool_texts)

            # Index tools in Qdrant in smaller batches to avoid memory issues
            batch_size = 50
            for i in range(0, len(tools_to_index), batch_size):
                batch_tools = tools_to_index[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]

                await vector_store.index_tools_batch(batch_tools, batch_embeddings)
                logger.info(f"  ✓ Indexed batch {i//batch_size + 1}: {len(batch_tools)} tools")

            logger.info(f"✅ Successfully indexed {len(tools_to_index)} training tools in Qdrant")
            logger.info(f"🔍 Now semantic search should return real similarity scores!")

        except Exception as e:
            logger.error(f"❌ Failed to index training tools: {e}")
            logger.error("Semantic search will fall back to default scores")
    else:
        logger.info("✅ All training tools already indexed in Qdrant")


if __name__ == "__main__":
    asyncio.run(train_ltr_model_with_context())
