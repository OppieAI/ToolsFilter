"""Pydantic models for API requests and responses.

We use OpenAI's message format types directly for compatibility with LiteLLM
and other libraries. This ensures we support all message formats correctly.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID, uuid4

# Since we're using LiteLLM which handles the conversion, we'll define
# a simplified message format that's compatible with both OpenAI and Anthropic
ChatMessage = Dict[str, Any]  # Will be validated by LiteLLM

# Tool Models following OpenAI's function calling format
class ToolFunction(BaseModel):
    """Tool function definition following OpenAI format."""
    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: Dict[str, Any] = Field(
        description="Parameters as JSON Schema",
        default_factory=lambda: {"type": "object", "properties": {}}
    )


class Tool(BaseModel):
    """Tool definition compatible with OpenAI and MCP standards."""
    type: Literal["function"] = Field(default="function")
    function: ToolFunction = Field(description="Function details")
    
    # Additional MCP-specific fields
    category: Optional[str] = Field(default="general", description="Tool category")
    
    @classmethod
    def from_mcp(cls, name: str, description: str, parameters: Optional[Dict] = None, category: str = "general"):
        """Create Tool from MCP-style definition."""
        return cls(
            type="function",
            function=ToolFunction(
                name=name,
                description=description,
                parameters=parameters or {"type": "object", "properties": {}}
            ),
            category=category
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "function",
                "function": {
                    "name": "grep",
                    "description": "Search for patterns in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Pattern to search"},
                            "file": {"type": "string", "description": "File to search in"}
                        },
                        "required": ["pattern"]
                    }
                },
                "category": "search"
            }
        }


# API Request/Response Models
class ToolFilterRequest(BaseModel):
    """Request to filter tools based on conversation."""
    messages: List[ChatMessage] = Field(
        description="Conversation messages in OpenAI/Anthropic format"
    )
    available_tools: List[Tool] = Field(
        description="List of available tools to filter"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="User identifier for personalization"
    )
    max_tools: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of tools to return"
    )
    include_reasoning: Optional[bool] = Field(
        default=False,
        description="Include reasoning for recommendations"
    )
    
    @validator("messages")
    def validate_messages(cls, v):
        """Ensure at least one message is provided."""
        if not v:
            raise ValueError("At least one message must be provided")
        return v
    
    @validator("available_tools")
    def validate_tools(cls, v):
        """Ensure at least one tool is provided."""
        if not v:
            raise ValueError("At least one tool must be provided")
        return v


class RecommendedTool(BaseModel):
    """Single recommended tool with metadata."""
    tool_name: str = Field(description="Name of the recommended tool")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the recommendation"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Explanation for the recommendation"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tool_name": "grep",
                "confidence": 0.95,
                "reasoning": "User wants to search for patterns in files"
            }
        }


class ToolFilterResponse(BaseModel):
    """Response containing filtered tools."""
    recommended_tools: List[RecommendedTool] = Field(
        description="List of recommended tools"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Response metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "recommended_tools": [
                    {
                        "tool_name": "grep",
                        "confidence": 0.95,
                        "reasoning": "Best for pattern searching"
                    },
                    {
                        "tool_name": "find",
                        "confidence": 0.85,
                        "reasoning": "Good for file discovery"
                    }
                ],
                "metadata": {
                    "processing_time_ms": 42,
                    "embedding_model": "voyage-2",
                    "total_tools_analyzed": 50
                }
            }
        }


# Health Check Models
class HealthStatus(BaseModel):
    """Health check response."""
    status: Literal["healthy", "unhealthy"] = Field(
        description="Service health status"
    )
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, bool] = Field(
        description="Individual service statuses"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )


# Error Models
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(description="Error type")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for debugging"
    )