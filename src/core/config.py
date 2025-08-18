"""Configuration settings for PTR Tool Filter."""

from typing import List, Optional, Any
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_env: str = Field(default="development", description="API environment")
    log_level: str = Field(default="INFO", description="Logging level")

    # Embedding Service
    voyage_api_key: Optional[str] = Field(default=None, description="Voyage AI API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")

    primary_embedding_model: str = Field(
        default="voyage-2",
        description="Primary embedding model"
    )
    primary_embedding_api_key: str = Field(
        description="API key for primary embedding model"
    )
    fallback_embedding_model: Optional[str] = Field(
        default=None,
        description="Fallback embedding model"
    )
    fallback_embedding_api_key: Optional[str] = Field(
        default=None,
        description="API key for fallback embedding model"
    )
    embedding_batch_size: int = Field(
        default=100,
        description="Batch size for embedding generation"
    )
    embedding_cache_ttl: int = Field(
        default=3600,
        description="TTL for embedding cache in seconds"
    )

    # Vector Database
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant REST port")
    qdrant_grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    qdrant_prefer_grpc: bool = Field(default=True, description="Use gRPC for better performance")
    qdrant_collection_name: str = Field(default="tools", description="Collection name for tools")

    # Redis Cache
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")

    # Tool Filtering
    max_tools_to_return: int = Field(
        default=10,
        description="Maximum number of tools to return"
    )
    primary_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for primary model"
    )
    fallback_similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold for fallback model"
    )
    enable_query_enhancement: bool = Field(
        default=True,
        description="Enable multi-query enhancement for better tool matching"
    )
    enable_two_stage_search: bool = Field(
        default=False,
        description="Enable optimized two-stage search for large collections"
    )
    two_stage_threshold: int = Field(
        default=1000,
        description="Collection size threshold to trigger two-stage search"
    )
    search_cache_size: int = Field(
        default=1000,
        description="Maximum number of cached search results"
    )
    search_cache_ttl: int = Field(
        default=300,
        description="TTL for cached search results in seconds"
    )
    
    # Hybrid Search Configuration
    enable_hybrid_search: bool = Field(
        default=True,
        description="Enable hybrid semantic + BM25 search"
    )
    hybrid_search_method: str = Field(
        default="weighted",
        description="Hybrid method: 'weighted' or 'rrf'"
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic scores in hybrid search"
    )
    bm25_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 scores in hybrid search"
    )
    
    # BM25 Configuration
    bm25_variant: str = Field(
        default="okapi",
        description="BM25 variant: 'okapi', 'plus', or 'l'"
    )
    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        le=3.0,
        description="BM25 term frequency saturation parameter"
    )
    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 length normalization parameter"
    )

    # Security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    enable_api_key: bool = Field(default=False, description="Enable API key authentication")
    api_keys: str = Field(default="", description="Comma-separated list of valid API keys")
    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], description="Allowed CORS origins")

    # Evaluation
    mlflow_tracking_uri: str = Field(default="file:./mlruns", description="MLflow tracking URI")
    phoenix_port: int = Field(default=6006, description="Phoenix port")

    @property
    def api_keys_list(self) -> List[str]:
        """Get API keys as a list."""
        if not self.api_keys or self.api_keys.strip() == "":
            return []
        return [key.strip() for key in self.api_keys.split(",") if key.strip()]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    def __init__(self, **kwargs):
        """Initialize settings with validation."""
        super().__init__(**kwargs)
        # Validate that fallback API key is provided if fallback model is specified
        if self.fallback_embedding_model and not self.fallback_embedding_api_key:
            raise ValueError("fallback_embedding_api_key is required when fallback_embedding_model is specified")

    @property
    def qdrant_url(self) -> str:
        """Get Qdrant URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def redis_url(self) -> str:
        """Get Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.api_env.lower() == "production"

    def get_embedding_dimension(self, model: Optional[str] = None) -> Optional[int]:
        """Get embedding dimension based on model from config.

        Returns None if model not found in config (should fetch at runtime).
        """
        model = model or self.primary_embedding_model
        dimensions = {
            "voyage/voyage-2": 1024,
            "voyage/voyage-3": 1024,  # voyage-3 has same dimension as voyage-2
            "voyage/voyage-3.5": 1024,  # voyage-3.5 also has same dimension
            "voyage/voyage-large-2": 1536,
            "openai/text-embedding-3-small": 1536,
            "openai/text-embedding-3-large": 3072,
            "openai/text-embedding-ada-002": 1536,
            "cohere/embed-english-v3.0": 1024,
            "cohere/embed-multilingual-v3.0": 1024,
            # Google Gemini embeddings
            "gemini/gemini-embedding-001": 3072,
            "gemini/models/text-embedding-004": 768,
            "gemini/text-embedding-preview-0409": 768,
            # Also support without provider prefix for backward compatibility
            "voyage-2": 1024,
            "voyage-3": 1024,
            "voyage-3.5": 1024,
            "voyage-large-2": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "gemini-embedding-001": 3072,
            "text-embedding-004": 768,
            "text-embedding-preview-0409": 768,
        }
        return dimensions.get(model, None)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
