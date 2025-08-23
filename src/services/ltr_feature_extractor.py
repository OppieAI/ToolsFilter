"""Learning to Rank feature extraction service.

This module is responsible ONLY for extracting features from query-tool pairs.
Follows Single Responsibility Principle - only handles feature extraction logic.
"""
import random
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import Counter
import Levenshtein

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    enable_similarity_features: bool = True
    enable_name_features: bool = True
    enable_description_features: bool = True
    enable_parameter_features: bool = True
    enable_query_features: bool = True
    enable_metadata_features: bool = True


class LTRFeatureExtractor:
    """
    Extracts features for Learning to Rank.

    Single Responsibility: Feature extraction from query-tool pairs.
    Does NOT handle training, scoring, or ranking.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config or FeatureConfig()
        self._query_type_patterns = self._compile_query_patterns()
        self._code_patterns = self._compile_code_patterns()

    def _compile_query_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for query type detection."""
        return {
            'search': re.compile(r'\b(search|find|look|locate|get)\b', re.I),
            'read': re.compile(r'\b(read|load|open|fetch|retrieve)\b', re.I),
            'write': re.compile(r'\b(write|save|store|create|update)\b', re.I),
            'execute': re.compile(r'\b(run|execute|call|invoke|perform)\b', re.I),
            'analyze': re.compile(r'\b(analyze|inspect|examine|check|validate)\b', re.I),
        }

    def _compile_code_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for code detection."""
        return [
            re.compile(r'```[\s\S]*?```'),  # Code blocks
            re.compile(r'`[^`]+`'),  # Inline code
            re.compile(r'\b(def|class|function|var|let|const)\b'),  # Keywords
            re.compile(r'[a-zA-Z_]\w*\([^)]*\)'),  # Function calls
            re.compile(r'[a-zA-Z_]\w*\.\w+'),  # Method calls
        ]

    def extract_features(
        self,
        query: str,
        tool: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Extract features for a single query-tool pair.

        Args:
            query: Search query
            tool: Tool information dictionary
            context: Optional context (scores from other models, metadata)

        Returns:
            Dictionary of feature name to value
        """
        features = {}
        context = context or {}

        # Extract basic tool information
        tool_name = self._get_tool_name(tool)
        tool_description = self._get_tool_description(tool)
        tool_parameters = self._get_tool_parameters(tool)

        # 1. Similarity features (from other models)
        if self.config.enable_similarity_features:
            features.update(self._extract_similarity_features(context))

        # 2. Name matching features
        if self.config.enable_name_features:
            features.update(self._extract_name_features(query, tool_name))

        # 3. Description features
        if self.config.enable_description_features:
            features.update(self._extract_description_features(
                query, tool_description
            ))

        # 4. Parameter features
        if self.config.enable_parameter_features:
            features.update(self._extract_parameter_features(
                query, tool_parameters
            ))

        # 5. Query features
        if self.config.enable_query_features:
            features.update(self._extract_query_features(query))

        # 6. Tool metadata features
        if self.config.enable_metadata_features:
            features.update(self._extract_metadata_features(tool, context))
        
        # 7. Query-tool interaction features (NEW - replaces constant domain features)
        # These features capture the relationship between query and tool, solving the
        # problem of constant features like 'domain_network' that don't vary with queries
        features.update(self._extract_query_tool_interaction_features(
            query, tool_name, tool_description, tool_parameters, context
        ))

        return features

    def extract_features_batch(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> np.ndarray:
        """
        Extract features for multiple tools.

        Args:
            query: Search query
            tools: List of tools
            contexts: Optional list of contexts for each tool

        Returns:
            Feature matrix (n_tools x n_features)
        """
        contexts = contexts or [{}] * len(tools)

        feature_dicts = []
        for tool, context in zip(tools, contexts):
            features = self.extract_features(query, tool, context)
            feature_dicts.append(features)

        # Convert to numpy array with consistent feature ordering
        if not feature_dicts:
            return np.array([])

        # Get all feature names
        all_features = set()
        for fd in feature_dicts:
            all_features.update(fd.keys())

        feature_names = sorted(all_features)

        # Create feature matrix
        feature_matrix = np.zeros((len(tools), len(feature_names)))
        for i, fd in enumerate(feature_dicts):
            for j, fname in enumerate(feature_names):
                feature_matrix[i, j] = fd.get(fname, 0.0)

        return feature_matrix

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # Generate dummy features to get all names
        dummy_tool = {
            "function": {
                "name": "test_api_function",
                "description": "test function for api calls",
                "parameters": {
                    "properties": {"param1": {"description": "test parameter"}},
                    "required": ["param1"]
                }
            }
        }
        dummy_context = {
            "semantic_score": 0.5,
            "bm25_score": 0.5,
            "cross_encoder_score": 0.5,
            "initial_rank": 1.0,
            "initial_score": 0.5,
            "top_score": 0.8
        }
        features = self.extract_features("get user data from api", dummy_tool, dummy_context)
        return sorted(features.keys())

    # --- Helper methods for tool information extraction ---

    def _get_tool_name(self, tool: Dict[str, Any]) -> str:
        """Extract tool name from various formats."""
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("name", "")
        elif "tool_name" in tool:
            return tool["tool_name"]
        elif "name" in tool:
            return tool["name"]
        return ""

    def _get_tool_description(self, tool: Dict[str, Any]) -> str:
        """Extract tool description."""
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("description", "")
        elif "description" in tool:
            return tool["description"]
        return ""

    def _get_tool_parameters(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tool parameters."""
        if "function" in tool and isinstance(tool["function"], dict):
            return tool["function"].get("parameters", {})
        elif "parameters" in tool:
            return tool["parameters"]
        return {}

    # --- Feature extraction methods ---

    def _extract_similarity_features(
        self, context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract similarity scores from context."""
        features = {}

        # Get scores from other models
        features['semantic_similarity'] = context.get('semantic_score', 0.0)
        features['bm25_score'] = context.get('bm25_score', 0.0)
        features['cross_encoder_score'] = context.get('cross_encoder_score', 0.0)

        # Combined score if available
        # features['hybrid_score'] = context.get('hybrid_score', 0.0)

        # Score statistics
        scores = [v for v in features.values() if v > 0]
        if scores:
            features['score_mean'] = np.mean(scores)
            features['score_std'] = np.std(scores)
            features['score_max'] = np.max(scores)
            features['score_min'] = np.min(scores)
        else:
            features['score_mean'] = 0.0
            features['score_std'] = 0.0
            features['score_max'] = 0.0
            features['score_min'] = 0.0

        return features

    def _extract_name_features(
        self, query: str, tool_name: str
    ) -> Dict[str, float]:
        """Extract name-based matching features."""
        features = {}

        if not tool_name:
            return {
                'exact_name_match': 0.0,
                'partial_name_match': 0.0,
                'name_in_query': 0.0,
                'query_in_name': 0.0,
                'name_edit_distance': 1.0,
                'name_length_ratio': 0.0,
            }

        query_lower = query.lower()
        name_lower = tool_name.lower()

        # Exact match
        features['exact_name_match'] = float(name_lower in query_lower)

        # Partial match (any word from name in query)
        name_words = set(name_lower.replace('_', ' ').split())
        query_words = set(query_lower.split())
        common_words = name_words & query_words
        features['partial_name_match'] = len(common_words) / len(name_words) if name_words else 0.0

        # Name in query
        features['name_in_query'] = float(name_lower in query_lower)

        # Query terms in name
        features['query_in_name'] = float(any(word in name_lower for word in query_words))

        # Edit distance (normalized)
        max_len = max(len(query_lower), len(name_lower))
        if max_len > 0:
            distance = Levenshtein.distance(query_lower, name_lower)
            features['name_edit_distance'] = 1.0 - (distance / max_len)
        else:
            features['name_edit_distance'] = 0.0

        # Length ratio
        features['name_length_ratio'] = min(len(tool_name), len(query)) / max(len(tool_name), len(query), 1)

        return features

    def _extract_description_features(
        self, query: str, description: str
    ) -> Dict[str, float]:
        """Extract description-based features."""
        features = {}

        if not description:
            return {
                'description_length': 0.0,
                'description_word_overlap': 0.0,
                'description_char_overlap': 0.0,
                'keyword_density': 0.0,
            }

        # Description length (normalized by log)
        features['description_length'] = np.log1p(len(description)) / 10.0

        # Word overlap (Jaccard similarity)
        query_words = set(query.lower().split())
        desc_words = set(description.lower().split())
        union = query_words | desc_words
        intersection = query_words & desc_words
        features['description_word_overlap'] = len(intersection) / len(union) if union else 0.0

        # Character n-gram overlap
        query_chars = set(query.lower())
        desc_chars = set(description.lower())
        char_union = query_chars | desc_chars
        char_intersection = query_chars & desc_chars
        features['description_char_overlap'] = len(char_intersection) / len(char_union) if char_union else 0.0

        # Keyword density
        keyword_count = sum(1 for word in query_words if word in description.lower())
        features['keyword_density'] = keyword_count / len(desc_words) if desc_words else 0.0

        return features

    def _extract_parameter_features(
        self, query: str, parameters: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract parameter-based features."""
        features = {}

        if not parameters:
            return {
                'num_parameters': 0.0,
                'num_required_params': 0.0,
                'num_optional_params': 0.0,
                'param_complexity': 0.0,
                'param_name_match': 0.0,
            }

        properties = parameters.get('properties', {})
        required = parameters.get('required', [])

        # Count parameters
        features['num_parameters'] = len(properties) / 10.0  # Normalize
        features['num_required_params'] = len(required) / 10.0
        features['num_optional_params'] = (len(properties) - len(required)) / 10.0

        # Parameter complexity (depth of nested structures)
        features['param_complexity'] = self._calculate_param_complexity(properties) / 10.0

        # Parameter name matching with query
        param_names = set(properties.keys())
        query_words = set(query.lower().split())
        matched = sum(1 for param in param_names if any(word in param.lower() for word in query_words))
        features['param_name_match'] = matched / len(param_names) if param_names else 0.0

        return features

    def _extract_query_features(self, query: str) -> Dict[str, float]:
        """Extract query-specific features."""
        features = {}

        # Query length
        features['query_length'] = len(query.split()) / 20.0  # Normalize
        features['query_char_length'] = len(query) / 100.0

        # Query type detection
        for qtype, pattern in self._query_type_patterns.items():
            features[f'query_type_{qtype}'] = float(bool(pattern.search(query)))

        # Code snippet detection
        features['has_code_snippet'] = float(
            any(pattern.search(query) for pattern in self._code_patterns)
        )

        # Special characters
        features['has_special_chars'] = float(bool(re.search(r'[^a-zA-Z0-9\s]', query)))

        # Question detection
        features['is_question'] = float(query.strip().endswith('?'))

        return features

    def _extract_metadata_features(
        self, tool: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract tool metadata features - keeping only non-constant features."""
        features = {}

        # Extract tool name for pattern analysis
        tool_name = tool.get('name', '')
        description = tool.get('description', '')
        name_lower = tool_name.lower()
        desc_lower = description.lower()

        # Tool specificity (longer, more specific names usually indicate specialized tools)
        features['name_specificity'] = min(len(tool_name) / 50.0, 1.0) if tool_name else 0.0  # Normalize to [0, 1]

        # Position features from context (initial ranking)
        features['initial_rank'] = context.get('initial_rank', 0.0) / 100.0  # Normalize
        features['initial_score'] = context.get('initial_score', 0.0)
        
        # Score difference from top result
        top_score = context.get('top_score', features['initial_score'])
        features['score_diff_from_top'] = max(0.0, top_score - features['initial_score'])

        return features

    def _extract_query_tool_interaction_features(
        self, 
        query: str, 
        tool_name: str, 
        tool_description: str, 
        tool_parameters: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract query-tool interaction features that capture relationships."""
        features = {}
        
        query_tokens = set(query.lower().split())
        tool_tokens = set(tool_name.lower().split('_')) if tool_name else set()
        desc_tokens = set(tool_description.lower().split()) if tool_description else set()
        
        # 1. Lexical overlap features
        if tool_tokens:
            features['query_tool_name_overlap'] = len(query_tokens & tool_tokens) / max(len(query_tokens), 1)
        else:
            features['query_tool_name_overlap'] = 0.0
            
        if desc_tokens:
            features['query_desc_overlap'] = len(query_tokens & desc_tokens) / max(len(query_tokens), 1)
            # Jaccard similarity
            union_tokens = query_tokens | desc_tokens
            features['jaccard_similarity'] = len(query_tokens & desc_tokens) / len(union_tokens) if union_tokens else 0.0
        else:
            features['query_desc_overlap'] = 0.0
            features['jaccard_similarity'] = 0.0
        
        # 2. Semantic features (using available similarity scores)
        # If semantic embeddings are available in context, use them
        # Otherwise, fall back to lexical similarity as proxy
        query_tool_sim = context.get('query_tool_cosine_sim', features.get('jaccard_similarity', 0.0))
        query_desc_sim = context.get('query_desc_cosine_sim', features.get('query_desc_overlap', 0.0))
        
        features['query_tool_cosine_sim'] = query_tool_sim
        features['query_desc_cosine_sim'] = query_desc_sim
        
        # 3. Intent matching - verb alignment
        query_verbs = self._extract_action_verbs(query)
        features['verb_match'] = float(any(verb in tool_name.lower() for verb in query_verbs)) if query_verbs else 0.0
        
        # Action alignment based on CRUD operations
        features['action_alignment'] = self._calculate_action_alignment(query, tool_name, tool_description)
        
        # 4. Length ratios
        features['query_length_ratio'] = len(query) / max(len(tool_description), 1) if tool_description else 0.0
        features['name_length_ratio'] = len(query) / max(len(tool_name), 1) if tool_name else 0.0
        
        # 5. Parameter matching
        features['param_relevance'] = self._calculate_param_relevance(query, tool_parameters)
        features['required_param_match'] = self._match_required_params(query, tool_parameters.get('required', []))
        
        return features
    
    def _extract_action_verbs(self, query: str) -> List[str]:
        """Extract action verbs from query."""
        # Common action verbs in API/tool contexts
        action_verbs = {
            'get', 'fetch', 'retrieve', 'find', 'search', 'list', 'show', 'read',
            'create', 'add', 'new', 'make', 'generate', 'build', 'post',
            'update', 'modify', 'edit', 'change', 'set', 'patch', 'put',
            'delete', 'remove', 'destroy', 'clear', 'drop',
            'send', 'upload', 'download', 'copy', 'move', 'sync',
            'validate', 'verify', 'check', 'test', 'parse', 'analyze'
        }
        
        query_words = query.lower().split()
        return [word for word in query_words if word in action_verbs]
    
    def _calculate_action_alignment(self, query: str, tool_name: str, tool_description: str) -> float:
        """Calculate how well query action aligns with tool action."""
        query_lower = query.lower()
        tool_text = (tool_name + ' ' + tool_description).lower()
        
        # CRUD alignment patterns (fixed: use tuples as keys, not lists)
        alignments = [
            # (query_patterns, tool_patterns)
            (
                ['get', 'fetch', 'retrieve', 'find', 'search', 'list', 'show'],
                ['get', 'fetch', 'retrieve', 'find', 'search', 'list', 'show', 'read']
            ),
            (
                ['create', 'add', 'new', 'make', 'generate', 'build'],
                ['create', 'add', 'new', 'make', 'generate', 'build', 'post']
            ),
            (
                ['update', 'modify', 'edit', 'change', 'set'],
                ['update', 'modify', 'edit', 'change', 'set', 'patch', 'put']
            ),
            (
                ['delete', 'remove', 'destroy', 'clear'],
                ['delete', 'remove', 'destroy', 'clear', 'drop']
            )
        ]
        
        max_alignment = 0.0
        for query_patterns, tool_patterns in alignments:
            query_match = any(pattern in query_lower for pattern in query_patterns)
            tool_match = any(pattern in tool_text for pattern in tool_patterns)
            if query_match and tool_match:
                max_alignment = 1.0
                break
            elif query_match or tool_match:
                max_alignment = max(max_alignment, 0.5)
        
        return max_alignment
    
    def _calculate_param_relevance(self, query: str, parameters: Dict[str, Any]) -> float:
        """Calculate how relevant tool parameters are to the query."""
        if not parameters or 'properties' not in parameters:
            return 0.0
        
        properties = parameters['properties']
        if not properties:
            return 0.0
        
        query_tokens = set(query.lower().split())
        
        relevant_params = 0
        for param_name, param_info in properties.items():
            param_text = param_name.lower()
            if isinstance(param_info, dict) and 'description' in param_info:
                param_text += ' ' + param_info['description'].lower()
            
            # Check if any query token appears in parameter name or description
            param_tokens = set(param_text.split())
            if query_tokens & param_tokens:
                relevant_params += 1
        
        return relevant_params / len(properties)
    
    def _match_required_params(self, query: str, required_params: List[str]) -> float:
        """Check how many required parameters are mentioned in query."""
        if not required_params:
            return 0.0
        
        query_lower = query.lower()
        matched_params = sum(1 for param in required_params if param.lower() in query_lower)
        
        return matched_params / len(required_params)
    
    def _calculate_param_complexity(self, properties: Dict[str, Any]) -> float:
        """Calculate complexity score for parameters."""
        def get_depth(obj, current_depth=0):
            if not isinstance(obj, dict):
                return current_depth
            if not obj:
                return current_depth
            return max(get_depth(v, current_depth + 1) for v in obj.values())

        return get_depth(properties)
