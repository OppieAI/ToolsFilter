"""Main FastAPI application for PTR Tool Filter."""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.core.config import get_settings
from src.core.models import HealthStatus, ErrorResponse
from src.api.endpoints import router
from src.services.vector_store import VectorStoreService
from src.services.embeddings import EmbeddingService
from src.services.search_service import SearchService

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global services
vector_store: VectorStoreService = None
fallback_vector_store: VectorStoreService = None
embedding_service: EmbeddingService = None
fallback_embedding_service: EmbeddingService = None
search_service: SearchService = None
fallback_search_service: SearchService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global vector_store, fallback_vector_store, embedding_service, fallback_embedding_service, search_service, fallback_search_service
    
    logger.info("Starting PTR Tool Filter API...")
    
    try:
        # Initialize primary embedding service
        embedding_service = EmbeddingService(
            model=settings.primary_embedding_model,
            api_key=settings.primary_embedding_api_key
        )
        
        # Initialize primary vector store
        primary_dimension = settings.get_embedding_dimension(settings.primary_embedding_model)
        if primary_dimension is None:
            logger.info(f"Model {settings.primary_embedding_model} not in config, fetching dimension...")
            primary_dimension = await embedding_service.get_embedding_dimension(settings.primary_embedding_model)
        
        logger.info(f"Primary model: {settings.primary_embedding_model} with dimension: {primary_dimension}")
        
        vector_store = VectorStoreService(
            embedding_dimension=primary_dimension,
            model_name=settings.primary_embedding_model,
            similarity_threshold=settings.primary_similarity_threshold
        )
        await vector_store.initialize()
        
        # Initialize primary search service
        search_service = SearchService(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        logger.info("Primary search service initialized")
        
        # Initialize fallback services if fallback model is configured
        if settings.fallback_embedding_model:
            # Initialize fallback embedding service
            fallback_embedding_service = EmbeddingService(
                model=settings.fallback_embedding_model,
                api_key=settings.fallback_embedding_api_key
            )
            
            fallback_dimension = settings.get_embedding_dimension(settings.fallback_embedding_model)
            if fallback_dimension is None:
                logger.info(f"Model {settings.fallback_embedding_model} not in config, fetching dimension...")
                fallback_dimension = await fallback_embedding_service.get_embedding_dimension()
            
            logger.info(f"Fallback model: {settings.fallback_embedding_model} with dimension: {fallback_dimension}")
            
            fallback_vector_store = VectorStoreService(
                embedding_dimension=fallback_dimension,
                model_name=settings.fallback_embedding_model,
                similarity_threshold=settings.fallback_similarity_threshold
            )
            await fallback_vector_store.initialize()
            
            # Initialize fallback search service
            fallback_search_service = SearchService(
                vector_store=fallback_vector_store,
                embedding_service=fallback_embedding_service
            )
            logger.info("Fallback search service initialized")
        
        logger.info("All services initialized successfully")
        
        yield
        
    finally:
        # Cleanup
        logger.info("Shutting down PTR Tool Filter API...")
        if vector_store:
            await vector_store.close()
        if fallback_vector_store:
            await fallback_vector_store.close()


# Create FastAPI app
app = FastAPI(
    title="PTR Tool Filter API",
    description="Precision-driven Tool Recommendation system for filtering MCP tools",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.api_env == "development" else settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    return response


# Include routers
app.include_router(router, prefix="/api/v1")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint."""
    return {
        "message": "PTR Tool Filter API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# Health check endpoint
@app.get("/health", response_model=HealthStatus, tags=["Health"])
async def health_check():
    """Check health status of all services."""
    services_status = {}
    
    # Check vector store
    try:
        await vector_store.health_check()
        services_status["vector_store"] = True
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        services_status["vector_store"] = False
    
    # Check Redis cache
    try:
        await embedding_service.cache_health_check()
        services_status["cache"] = True
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        services_status["cache"] = False
    
    # Check embedding service
    services_status["embedding_service"] = embedding_service is not None
    
    # Determine overall status
    all_healthy = all(services_status.values())
    
    return HealthStatus(
        status="healthy" if all_healthy else "unhealthy",
        services=services_status
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred",
            details={"error": str(exc)} if settings.api_env == "development" else None
        ).dict()
    )


def get_app() -> FastAPI:
    """Get FastAPI application instance."""
    return app


def get_vector_store() -> VectorStoreService:
    """Get vector store service instance."""
    return vector_store


def get_fallback_vector_store() -> VectorStoreService:
    """Get fallback vector store service instance."""
    return fallback_vector_store


def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return embedding_service


def get_fallback_embedding_service() -> EmbeddingService:
    """Get fallback embedding service instance."""
    return fallback_embedding_service


def get_search_service() -> SearchService:
    """Get search service instance."""
    return search_service


def get_fallback_search_service() -> SearchService:
    """Get fallback search service instance."""
    return fallback_search_service


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_env == "development",
        log_level=settings.log_level.lower()
    )