"""Tests for the enhanced embedding functionality."""

import pytest
from src.services.embedding_enhancer import ToolEmbeddingEnhancer


class TestToolEmbeddingEnhancer:
    """Test cases for ToolEmbeddingEnhancer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = ToolEmbeddingEnhancer()
        
        # Simple tool example
        self.simple_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or zip code"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        # Complex tool example
        self.complex_tool = {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search for records in a database using SQL queries",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL query to execute"
                        },
                        "database": {
                            "type": "string",
                            "description": "Name of the database to search"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Query timeout in seconds"
                        }
                    },
                    "required": ["query", "database"]
                }
            }
        }
    
    def test_simple_tool_enhancement(self):
        """Test enhancement of a simple tool."""
        result = self.enhancer.tool_to_rich_text(self.simple_tool)
        
        # Check that key information is present
        assert "get_weather" in result
        assert "weather" in result.lower()
        assert "location" in result
        assert "string" in result
        assert "[required]" in result
        assert "City name or zip code" in result
        
        # Check that it's more informative than simple concatenation
        simple_text = "get_weather: Get current weather for a location"
        assert len(result) > len(simple_text)
    
    def test_complex_tool_enhancement(self):
        """Test enhancement of a complex tool with multiple parameters."""
        result = self.enhancer.tool_to_rich_text(self.complex_tool)
        
        # Check all parameters are captured
        assert "query" in result
        assert "database" in result
        assert "limit" in result
        assert "timeout" in result
        
        # Check parameter types
        assert "string" in result
        assert "integer" in result
        
        # Check required vs optional distinction
        assert result.count("[required]") == 2  # query and database
        assert result.count("[optional]") == 2  # limit and timeout
        
        # Check parameter descriptions are included
        assert "SQL query to execute" in result
        assert "Query timeout in seconds" in result
        
        # Check parameter counts
        assert "Total parameters: 4" in result
        assert "Required parameters: 2" in result
    
    def test_keyword_extraction(self):
        """Test keyword extraction for better matching."""
        # Test database tool
        db_result = self.enhancer.tool_to_rich_text(self.complex_tool)
        assert "Keywords:" in db_result
        assert any(kw in db_result.lower() for kw in ["sql", "database", "query"])
        
        # Test git tool
        git_tool = {
            "type": "function",
            "function": {
                "name": "git_commit",
                "description": "Create a git commit with staged changes"
            }
        }
        git_result = self.enhancer.tool_to_rich_text(git_tool)
        assert any(kw in git_result.lower() for kw in ["git", "commit", "version"])
    
    def test_tool_without_parameters(self):
        """Test enhancement of a tool without parameters."""
        no_param_tool = {
            "type": "function",
            "function": {
                "name": "get_timestamp",
                "description": "Get the current Unix timestamp",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
        
        result = self.enhancer.tool_to_rich_text(no_param_tool)
        assert "get_timestamp" in result
        assert "Unix timestamp" in result
        assert "Parameters: none" in result or "Total parameters: 0" in result
    
    def test_tool_with_category_and_tags(self):
        """Test enhancement of a tool with category and tags."""
        tagged_tool = {
            "type": "function",
            "function": {
                "name": "query_api",
                "description": "Query an external API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint": {"type": "string"}
                    }
                },
                "category": "network",
                "tags": ["api", "http", "rest"]
            }
        }
        
        result = self.enhancer.tool_to_rich_text(tagged_tool)
        assert "Category: network" in result
        assert "Tags:" in result
        assert "api" in result
        assert "http" in result
    
    def test_batch_enhancement(self):
        """Test batch enhancement of multiple tools."""
        tools = [self.simple_tool, self.complex_tool]
        results = self.enhancer.batch_tool_to_rich_text(tools)
        
        assert len(results) == 2
        assert "get_weather" in results[0]
        assert "search_database" in results[1]
    
    def test_non_function_tool_fallback(self):
        """Test handling of non-function type tools."""
        non_function_tool = {
            "name": "simple_tool",
            "description": "A simple non-function tool"
        }
        
        result = self.enhancer.tool_to_rich_text(non_function_tool)
        assert "simple_tool" in result
        assert "simple non-function tool" in result
    
    def test_information_gain(self):
        """Test that enhanced text contains significantly more information."""
        # Original simple approach
        def old_tool_to_text(tool):
            if tool.get("type") == "function":
                function = tool.get("function", {})
                name = function.get("name", "")
                description = function.get("description", "")
                return f"{name}: {description}"
            return str(tool)
        
        old_text = old_tool_to_text(self.complex_tool)
        new_text = self.enhancer.tool_to_rich_text(self.complex_tool)
        
        # Enhanced text should be at least 3x longer due to parameter information
        assert len(new_text) > len(old_text) * 3
        
        # Enhanced text should contain information not in the simple version
        assert "required" in new_text.lower()
        assert "optional" in new_text.lower()
        assert "integer" in new_text.lower()
        assert "SQL query to execute" in new_text