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

# JSON Schema Models for Tool Parameters
class ParameterProperty(BaseModel):
    """Individual parameter property definition."""
    type: str = Field(description="JSON Schema type (string, number, integer, boolean, array, object)")
    description: Optional[str] = Field(default=None, description="Description of the parameter")
    enum: Optional[List[Union[str, int, float, bool]]] = Field(default=None, description="Allowed values")
    default: Optional[Any] = Field(default=None, description="Default value")
    # For arrays
    items: Optional[Dict[str, Any]] = Field(default=None, description="Schema for array items")
    # For numbers
    minimum: Optional[Union[int, float]] = Field(default=None, description="Minimum value")
    maximum: Optional[Union[int, float]] = Field(default=None, description="Maximum value")
    # For strings
    minLength: Optional[int] = Field(default=None, description="Minimum string length")
    maxLength: Optional[int] = Field(default=None, description="Maximum string length")
    pattern: Optional[str] = Field(default=None, description="Regex pattern for validation")


class ToolParameters(BaseModel):
    """Tool parameters following JSON Schema specification."""
    type: Literal["object"] = Field(default="object", description="Parameters are always an object")
    properties: Dict[str, Union[ParameterProperty, Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Parameter properties"
    )
    required: List[str] = Field(
        default_factory=list,
        description="List of required parameter names"
    )
    additionalProperties: bool = Field(
        default=False,
        description="Whether additional properties are allowed"
    )
    
    @validator("required")
    def validate_required(cls, v, values):
        """Ensure required parameters exist in properties."""
        if "properties" in values:
            props = values["properties"]
            for req in v:
                if req not in props:
                    raise ValueError(f"Required parameter '{req}' not found in properties")
        return v


# Tool Models following OpenAI's function calling format
class Tool(BaseModel):
    """Tool definition compatible with OpenAI function calling format."""
    type: Literal["function"] = Field(default="function")
    name: str = Field(description="Function name")
    description: str = Field(description="Function description")
    parameters: Union[ToolParameters, Dict[str, Any]] = Field(
        description="Parameters as JSON Schema",
        default_factory=lambda: ToolParameters()
    )
    strict: Optional[bool] = Field(default=True, description="Enforce strict schema validation")
    
    # Additional fields for compatibility
    category: Optional[str] = Field(default="general", description="Tool category")
    
    @classmethod
    def from_mcp(cls, name: str, description: str, parameters: Optional[Dict] = None, category: str = "general"):
        """Create Tool from MCP-style definition."""
        params = parameters or {"type": "object", "properties": {}}
        if "additionalProperties" not in params:
            params["additionalProperties"] = False
        if "required" not in params:
            params["required"] = []
        
        return cls(
            type="function",
            name=name,
            description=description,
            parameters=params,
            strict=True,
            category=category
        )
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "function",
                "name": "get_weather",
                "description": "Retrieves current weather for the given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia"
                        },
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Units the temperature will be returned in."
                        }
                    },
                    "required": ["location", "units"],
                    "additionalProperties": False
                },
                "strict": True
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