.PHONY: help build up down logs shell test lint format clean

# Default target
help:
	@echo "Available commands:"
	@echo "  make build       - Build Docker images"
	@echo "  make up          - Start all services"
	@echo "  make up-dev      - Start all services in development mode"
	@echo "  make down        - Stop all services"
	@echo "  make logs        - View logs from all services"
	@echo "  make shell       - Open shell in API container"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linting"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean up volumes and cache"

# Build Docker images
build:
	docker compose build

# Start services in production mode
up:
	docker compose up -d

# Start services in development mode
up-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

down-dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

restart-api:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml restart api

flush-redis:
	docker exec ptr_redis redis-cli FLUSHALL

# Stop all services
down:
	docker compose down

# View logs
logs:
	docker compose logs -f

# Open shell in API container
shell:
	docker compose exec api /bin/bash

# Run tests
test:
	docker compose exec api pytest tests/ -v

# Run linting
lint:
	docker compose exec api ruff check src/

# Format code
format:
	docker compose exec api black src/

# Clean up
clean:
	docker compose down -v
	rm -rf qdrant_storage/ redis_data/ redisinsight/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development shortcuts
dev: up-dev
prod: up
