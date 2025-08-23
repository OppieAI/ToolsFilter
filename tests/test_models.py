"""Tests for core models."""

import pytest
from src.core.models import Tool, ToolFilterRequest, ToolFilterResponse, RecommendedTool


class TestModels:
    """Test cases for Pydantic models."""
    
    def test_tool_from_mcp(self):
        """Test creating Tool from MCP format."""
        tool = Tool.from_mcp(
            name="grep",
            description="Search for patterns",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"}
                },
                "required": ["pattern"]
            }
        )
        
        assert tool.type == "function"
        assert tool.name == "grep"
        assert tool.description == "Search for patterns"
        assert tool.parameters == {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"}
            },
            "required": ["pattern"],
            "additionalProperties": False
        }
        assert tool.strict == True
    
    def test_tool_filter_request_validation(self):
        """Test ToolFilterRequest validation."""
        # Valid request
        request = ToolFilterRequest(
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            available_tools=[
                Tool(
                    type="function",
                    name="test",
                    description="Test tool",
                    parameters={"type": "object", "properties": {}}
                )
            ]
        )
        
        assert len(request.messages) == 1
        assert len(request.available_tools) == 1
        assert request.max_tools == 10  # default
        assert request.include_reasoning is False  # default
    
    def test_tool_filter_request_invalid_messages(self):
        """Test ToolFilterRequest with invalid messages."""
        with pytest.raises(ValueError):
            ToolFilterRequest(
                messages=[],  # Empty messages
                available_tools=[
                    Tool(
                        type="function",
                        name="test",
                        description="Test tool",
                        parameters={"type": "object", "properties": {}}
                    )
                ]
            )
    
    def test_recommended_tool(self):
        """Test RecommendedTool model."""
        tool = RecommendedTool(
            tool_name="find",
            confidence=0.95,
            reasoning="High relevance to search intent"
        )
        
        assert tool.tool_name == "find"
        assert tool.confidence == 0.95
        assert tool.reasoning == "High relevance to search intent"
    
    def test_tool_filter_response(self):
        """Test ToolFilterResponse model."""
        response = ToolFilterResponse(
            recommended_tools=[
                RecommendedTool(
                    tool_name="grep",
                    confidence=0.85
                )
            ],
            metadata={
                "processing_time_ms": 42.5,
                "total_tools": 10
            }
        )
        
        assert len(response.recommended_tools) == 1
        assert response.recommended_tools[0].tool_name == "grep"
        assert response.metadata["processing_time_ms"] == 42.5