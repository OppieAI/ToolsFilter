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

    # Cross-Encoder Configuration
    enable_cross_encoder: bool = Field(
        default=True,
        description="Enable cross-encoder reranking for improved accuracy"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    cross_encoder_batch_size: int = Field(
        default=32,
        description="Batch size for cross-encoder inference"
    )
    cross_encoder_cache_size: int = Field(
        default=1000,
        description="Cache size for cross-encoder predictions"
    )
    cross_encoder_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight for cross-encoder scores when combining with original scores"
    )
    cross_encoder_top_k: int = Field(
        default=30,
        description="Number of candidates to rerank with cross-encoder"
    )
    
    # Two-Stage Filtering Configuration
    enable_two_stage_filtering: bool = Field(
        default=False,
        description="Enable two-stage filtering: aggressive first stage + precise reranking"
    )
    two_stage_stage1_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Stage 1 threshold: lower threshold to cast wider net"
    )
    two_stage_stage1_limit: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Stage 1 limit: higher limit for initial candidate retrieval"
    )
    two_stage_stage2_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Stage 2 threshold: stricter threshold for precise filtering"
    )
    two_stage_enable_confidence_cutoff: bool = Field(
        default=True,
        description="Apply confidence-based result limiting in stage 2"
    )

    # LTR (Learning to Rank) Configuration
    enable_ltr: bool = Field(
        default=True,
        description="Enable Learning to Rank model (disabled by default until trained)"
    )
    ltr_model_path: str = Field(
        default="./models/ltr_xgboost",
        description="Path to trained LTR model"
    )
    ltr_objective: str = Field(
        default="rank:pairwise",
        description="LTR ranking objective: rank:pairwise, rank:ndcg, or rank:map"
    )
    ltr_cache_predictions: bool = Field(
        default=True,
        description="Cache LTR predictions for performance"
    )
    ltr_cache_size: int = Field(
        default=1000,
        description="Size of LTR prediction cache"
    )
    ltr_retrain_threshold: int = Field(
        default=1000,
        description="Number of new evaluations before triggering retraining"
    )

    # LTR Feature Configuration
    ltr_similarity_features: bool = Field(
        default=True,
        description="Include similarity scores in LTR features"
    )
    ltr_name_features: bool = Field(
        default=True,
        description="Include name-based features in LTR"
    )
    ltr_description_features: bool = Field(
        default=True,
        description="Include description-based features in LTR"
    )
    ltr_parameter_features: bool = Field(
        default=True,
        description="Include parameter-based features in LTR"
    )
    ltr_query_features: bool = Field(
        default=True,
        description="Include query-based features in LTR"
    )
    ltr_metadata_features: bool = Field(
        default=True,
        description="Include metadata features in LTR"
    )

    # LTR Training Configuration
    ltr_learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Learning rate for LTR model training"
    )
    ltr_max_depth: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum tree depth for XGBoost"
    )
    ltr_n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of trees in XGBoost ensemble"
    )
    ltr_subsample: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Subsample ratio for training instances"
    )
    ltr_colsample_bytree: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Feature sampling ratio per tree"
    )
    ltr_min_child_weight: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Minimum child weight for tree splits"
    )
    ltr_gamma: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="Minimum loss reduction for splits"
    )
    ltr_reg_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=10.0,
        description="L1 regularization term"
    )
    ltr_reg_lambda: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="L2 regularization term"
    )
    ltr_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    ltr_n_jobs: int = Field(
        default=-1,
        description="Number of parallel threads (-1 for all cores)"
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
