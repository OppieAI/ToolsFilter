"""Enhanced tool embedding service for richer semantic representation."""

from typing import Dict, Any, List, Optional
import re


class ToolEmbeddingEnhancer:
    """Enhances tool text representation for better embedding quality."""
    
    def __init__(self):
        """Initialize the embedding enhancer."""
        # Common tool patterns for keyword extraction
        self.patterns = {
            "search": ["find", "locate", "query", "lookup", "discover", "explore"],
            "file": ["fs", "directory", "path", "folder", "document", "storage"],
            "database": ["db", "sql", "query", "table", "record", "schema"],
            "api": ["http", "rest", "endpoint", "request", "webhook", "graphql"],
            "git": ["version", "commit", "branch", "repository", "merge", "pull"],
            "data": ["json", "xml", "csv", "parse", "transform", "convert"],
            "auth": ["login", "oauth", "token", "permission", "credential", "access"],
            "cloud": ["aws", "azure", "gcp", "deploy", "kubernetes", "docker"],
            "monitor": ["log", "metric", "alert", "trace", "debug", "observe"],
            "test": ["unit", "integration", "mock", "assert", "validate", "verify"]
        }
    
    def tool_to_rich_text(self, tool: Any) -> str:
        """
        Create rich text representation for better embeddings.
        
        This captures:
        - Name and description
        - Parameters with types and descriptions
        - Required vs optional parameters
        - Category/tags
        - Keywords for better matching
        
        Args:
            tool: Tool dictionary or Pydantic model with OpenAI/Claude function format
            
        Returns:
            Rich text representation of the tool
        """
        parts = []
        
        # Convert Pydantic model to dict if necessary
        if hasattr(tool, 'dict'):
            tool = tool.dict()
        elif hasattr(tool, 'model_dump'):
            tool = tool.model_dump()
        
        if tool.get("type") == "function":
            # Handle both old nested and new flat structure
            if "function" in tool:
                # Old nested structure
                function = tool.get("function", {})
                name = function.get("name", "")
                description = function.get("description", "")
                params = function.get("parameters", {})
            else:
                # New flat structure
                name = tool.get("name", "")
                description = tool.get("description", "")
                params = tool.get("parameters", {})
            
            # 1. Name and description
            if name:
                parts.append(f"Tool: {name}")
            if description:
                parts.append(f"Description: {description}")
            
            # 2. Parameters with types and descriptions
            if params.get("properties"):
                param_texts = []
                required_params = params.get("required", [])
                
                for param_name, param_spec in params["properties"].items():
                    param_type = param_spec.get("type", "any")
                    param_desc = param_spec.get("description", "")
                    is_required = param_name in required_params
                    
                    # Build parameter text
                    param_text = f"{param_name} ({param_type})"
                    if param_desc:
                        param_text += f": {param_desc}"
                    if is_required:
                        param_text += " [required]"
                    else:
                        param_text += " [optional]"
                    
                    param_texts.append(param_text)
                
                if param_texts:
                    parts.append(f"Parameters: {', '.join(param_texts)}")
                    
                # Add parameter count information
                parts.append(f"Total parameters: {len(params['properties'])}")
                parts.append(f"Required parameters: {len(required_params)}")
            else:
                parts.append("Parameters: none")
            
            # 3. Category/tags (if available in tool metadata)
            category = tool.get("category", "")
            if category:
                parts.append(f"Category: {category}")
            
            tags = tool.get("tags", [])
            if tags:
                if isinstance(tags, list):
                    parts.append(f"Tags: {', '.join(tags)}")
                else:
                    parts.append(f"Tags: {tags}")
            
            # 4. Examples (if available)
            examples = tool.get("examples", "")
            if examples:
                if isinstance(examples, list):
                    parts.append(f"Examples: {'; '.join(examples)}")
                else:
                    parts.append(f"Examples: {examples}")
            
            # 5. Extract and add keywords
            keywords = self._extract_keywords(name, description)
            if keywords:
                parts.append(f"Keywords: {', '.join(keywords)}")
            
            # 6. Add return type information if available
            returns = tool.get("returns", {})
            if returns:
                return_type = returns.get("type", "")
                return_desc = returns.get("description", "")
                if return_type:
                    return_text = f"Returns: {return_type}"
                    if return_desc:
                        return_text += f" - {return_desc}"
                    parts.append(return_text)
        else:
            # Fallback for non-function tools
            name = tool.get("name", "")
            description = tool.get("description", "")
            if name:
                parts.append(f"Tool: {name}")
            if description:
                parts.append(f"Description: {description}")
            
            # Still try to extract keywords
            if name or description:
                keywords = self._extract_keywords(name, description)
                if keywords:
                    parts.append(f"Keywords: {', '.join(keywords)}")
        
        # Join all parts with separator
        return " | ".join(parts) if parts else str(tool)
    
    def _extract_keywords(self, name: str, description: str) -> List[str]:
        """
        Extract relevant keywords for better matching.
        
        Args:
            name: Tool name
            description: Tool description
            
        Returns:
            List of extracted keywords (max 10)
        """
        keywords = set()
        
        # Combine name and description for analysis
        text = f"{name} {description}".lower()
        
        # Check against known patterns
        for pattern, aliases in self.patterns.items():
            # Check if pattern appears in text
            if pattern in text:
                keywords.add(pattern)
            
            # Check for aliases
            for alias in aliases:
                if alias in text and len(alias) > 2:  # Skip very short aliases
                    keywords.add(alias)
                    if len(keywords) >= 10:
                        return sorted(list(keywords))[:10]
        
        # Extract potential technical terms (camelCase, snake_case, kebab-case)
        # CamelCase
        camel_pattern = r'[A-Z][a-z]+(?=[A-Z])|[A-Z][a-z]+'
        camel_words = re.findall(camel_pattern, name)
        for word in camel_words[:3]:  # Limit to 3 camelCase words
            if len(word) > 2:
                keywords.add(word.lower())
        
        # snake_case and kebab-case components
        snake_parts = name.split('_')
        kebab_parts = name.split('-')
        for parts in [snake_parts, kebab_parts]:
            for part in parts[:3]:  # Limit to 3 parts
                if len(part) > 2:
                    keywords.add(part.lower())
        
        # Common action verbs in tool names
        action_verbs = ["get", "set", "create", "delete", "update", "list", "fetch", 
                       "send", "receive", "process", "analyze", "compute", "calculate"]
        for verb in action_verbs:
            if verb in text:
                keywords.add(verb)
                if len(keywords) >= 10:
                    return sorted(list(keywords))[:10]
        
        return sorted(list(keywords))[:10]
    
    def batch_tool_to_rich_text(self, tools: List[Any]) -> List[str]:
        """
        Convert multiple tools to rich text representations.
        
        Args:
            tools: List of tool dictionaries or Pydantic models
            
        Returns:
            List of rich text representations
        """
        return [self.tool_to_rich_text(tool) for tool in tools]


def create_enhanced_tool_text(tool: Any) -> str:
    """
    Convenience function to create enhanced tool text.
    
    Args:
        tool: Tool dictionary or Pydantic model
        
    Returns:
        Rich text representation
    """
    enhancer = ToolEmbeddingEnhancer()
    return enhancer.tool_to_rich_text(tool)