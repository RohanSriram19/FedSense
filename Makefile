# FedSense Makefile
# Production-quality federated learning system for wearable time-series anomaly detection

# Configuration
PYTHON_VERSION := 3.11
PROJECT_NAME := fedsense
VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip
UVICORN := $(VENV_NAME)/bin/uvicorn
PYTEST := $(VENV_NAME)/bin/pytest

# Docker services
DOCKER_COMPOSE := docker compose
TRITON_SERVICE := triton-inference-server
MLFLOW_SERVICE := mlflow-server

# Data and model paths
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs
RESULTS_DIR := results

# Default target
.PHONY: help
help: ## Show this help message
	@echo "FedSense - Federated Time-Series Anomaly Detection for Wearables"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
.PHONY: venv
venv: ## Create Python virtual environment and install dependencies
	@echo "Creating virtual environment with Python $(PYTHON_VERSION)..."
	python$(PYTHON_VERSION) -m venv $(VENV_NAME)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	@echo "✓ Virtual environment created at $(VENV_NAME)"
	@echo "Activate with: source $(VENV_NAME)/bin/activate"

.PHONY: install
install: venv ## Install project dependencies (alias for venv)

.PHONY: clean-venv
clean-venv: ## Remove virtual environment
	rm -rf $(VENV_NAME)
	@echo "✓ Virtual environment removed"

# Code quality
.PHONY: lint
lint: ## Run linting with ruff
	@echo "Running linter..."
	$(PYTHON) -m ruff check fedsense/ tests/
	@echo "✓ Linting complete"

.PHONY: format
format: ## Format code with ruff and black
	@echo "Formatting code..."
	$(PYTHON) -m ruff format fedsense/ tests/
	$(PYTHON) -m black fedsense/ tests/ --line-length 88
	@echo "✓ Code formatting complete"

.PHONY: type-check
type-check: ## Run type checking with mypy
	@echo "Running type checker..."
	$(PYTHON) -m mypy fedsense/ --ignore-missing-imports
	@echo "✓ Type checking complete"

.PHONY: check
check: lint type-check ## Run all code quality checks
	@echo "✓ All quality checks passed"

# Testing
.PHONY: test
test: ## Run all tests with pytest
	@echo "Running tests..."
	$(PYTEST) tests/ -v --tb=short
	@echo "✓ All tests passed"

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	$(PYTEST) tests/ --cov=fedsense --cov-report=html --cov-report=term
	@echo "✓ Coverage report generated in htmlcov/"

.PHONY: test-fast
test-fast: ## Run tests without slow integration tests
	@echo "Running fast tests..."
	$(PYTEST) tests/ -v -m "not slow" --tb=short
	@echo "✓ Fast tests complete"

# Data management
.PHONY: data
data: ## Generate synthetic wearable datasets
	@echo "Generating synthetic datasets..."
	mkdir -p $(DATA_DIR)
	$(PYTHON) -c "
from fedsense.datasets import generate_synthetic_data, create_federated_splits
import pandas as pd
import os

print('Generating main dataset...')
df = generate_synthetic_data(
    n_samples=50000,
    anomaly_rate=0.05,
    random_seed=42
)
df.to_parquet('$(DATA_DIR)/wearable_data.parquet')
print(f'Saved {len(df)} samples to $(DATA_DIR)/wearable_data.parquet')

print('Creating federated splits...')
client_dfs = create_federated_splits(df, n_clients=5, alpha=0.5, random_seed=42)
for i, client_df in enumerate(client_dfs):
    client_df.to_parquet(f'$(DATA_DIR)/client_{i}_data.parquet')
    print(f'Client {i}: {len(client_df)} samples ({client_df[\"label\"].mean():.1%} anomalies)')

print('✓ Datasets generated successfully')
	"

.PHONY: data-stats
data-stats: ## Show dataset statistics
	@echo "Dataset Statistics:"
	$(PYTHON) -c "
import pandas as pd
import os

if os.path.exists('$(DATA_DIR)/wearable_data.parquet'):
    df = pd.read_parquet('$(DATA_DIR)/wearable_data.parquet')
    print(f'Main dataset: {len(df)} samples, {df[\"label\"].mean():.1%} anomalies')
    print(f'Duration: {(df[\"timestamp\"].max() - df[\"timestamp\"].min()).total_seconds():.0f} seconds')
    print()
    
    # Client statistics
    for i in range(5):
        client_file = f'$(DATA_DIR)/client_{i}_data.parquet'
        if os.path.exists(client_file):
            client_df = pd.read_parquet(client_file)
            print(f'Client {i}: {len(client_df)} samples, {client_df[\"label\"].mean():.1%} anomalies')
else:
    print('No datasets found. Run: make data')
	"

.PHONY: clean-data
clean-data: ## Remove generated datasets
	rm -rf $(DATA_DIR)/*.parquet
	@echo "✓ Generated datasets removed"

# Model training
.PHONY: train-local
train-local: ## Train model locally (centralized baseline)
	@echo "Training local baseline model..."
	mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	$(PYTHON) fedsense/train_local_jax.py \
		--data-path $(DATA_DIR)/wearable_data.parquet \
		--output-dir $(MODELS_DIR)/local_baseline \
		--epochs 50 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--window-length 250 \
		--hidden-dims 64 32 \
		--dropout-rate 0.2 \
		--seed 42
	@echo "✓ Local training complete"

.PHONY: train-local-dp
train-local-dp: ## Train model locally with differential privacy
	@echo "Training local model with differential privacy..."
	mkdir -p $(MODELS_DIR) $(LOGS_DIR)
	$(PYTHON) fedsense/train_local_jax.py \
		--data-path $(DATA_DIR)/wearable_data.parquet \
		--output-dir $(MODELS_DIR)/local_dp \
		--epochs 50 \
		--batch-size 64 \
		--learning-rate 0.001 \
		--window-length 250 \
		--hidden-dims 64 32 \
		--dropout-rate 0.2 \
		--use-dp \
		--noise-multiplier 1.0 \
		--l2-norm-clip 1.0 \
		--seed 42
	@echo "✓ Local DP training complete"

# Federated learning
.PHONY: fl-server
fl-server: ## Start federated learning server
	@echo "Starting federated learning server..."
	mkdir -p $(LOGS_DIR)
	$(PYTHON) fedsense/fl_server.py \
		--num-rounds 10 \
		--min-clients 3 \
		--sample-fraction 0.8 \
		--min-eval-clients 3 \
		--server-address 0.0.0.0:8080 \
		--output-dir $(MODELS_DIR)/federated \
		--config-path fedsense/config.py

.PHONY: fl-client
fl-client: ## Start single federated learning client (specify CLIENT_ID=0)
	@echo "Starting federated learning client $(CLIENT_ID)..."
	$(PYTHON) fedsense/fl_client.py \
		--client-id $(or $(CLIENT_ID),0) \
		--server-address 127.0.0.1:8080 \
		--data-path $(DATA_DIR)/client_$(or $(CLIENT_ID),0)_data.parquet \
		--epochs 3 \
		--batch-size 32 \
		--learning-rate 0.001 \
		--window-length 250 \
		--hidden-dims 64 32 \
		--dropout-rate 0.2

.PHONY: fl-clients
fl-clients: ## Start multiple federated clients (background processes)
	@echo "Starting 5 federated learning clients..."
	@for i in 0 1 2 3 4; do \
		echo "Starting client $$i..."; \
		$(PYTHON) fedsense/fl_client.py \
			--client-id $$i \
			--server-address 127.0.0.1:8080 \
			--data-path $(DATA_DIR)/client_$$i\_data.parquet \
			--epochs 3 \
			--batch-size 32 \
			--learning-rate 0.001 \
			--window-length 250 \
			--hidden-dims 64 32 \
			--dropout-rate 0.2 & \
	done
	@echo "✓ All clients started in background"

.PHONY: fl-stop
fl-stop: ## Stop all federated learning processes
	@echo "Stopping federated learning processes..."
	pkill -f "fl_server.py" || true
	pkill -f "fl_client.py" || true
	@echo "✓ FL processes stopped"

# Model export and serving
.PHONY: export-onnx
export-onnx: ## Export trained model to ONNX format
	@echo "Exporting model to ONNX..."
	mkdir -p $(MODELS_DIR)/onnx
	$(PYTHON) fedsense/export_onnx.py \
		--jax-model-path $(MODELS_DIR)/local_baseline/model.pkl \
		--output-path $(MODELS_DIR)/onnx/fedsense_model.onnx \
		--input-shape 1 250 4 \
		--opset-version 13
	@echo "✓ ONNX export complete"

.PHONY: triton-setup
triton-setup: export-onnx ## Setup Triton model repository
	@echo "Setting up Triton model repository..."
	mkdir -p docker/triton/models/fedsense/1
	cp $(MODELS_DIR)/onnx/fedsense_model.onnx docker/triton/models/fedsense/1/model.onnx
	@echo "✓ Triton model repository ready"

.PHONY: triton-up
triton-up: triton-setup ## Start Triton Inference Server
	@echo "Starting Triton Inference Server..."
	cd docker && $(DOCKER_COMPOSE) up -d $(TRITON_SERVICE)
	@echo "✓ Triton server starting... (check logs: make triton-logs)"

.PHONY: triton-down
triton-down: ## Stop Triton Inference Server
	@echo "Stopping Triton Inference Server..."
	cd docker && $(DOCKER_COMPOSE) down $(TRITON_SERVICE)
	@echo "✓ Triton server stopped"

.PHONY: triton-logs
triton-logs: ## Show Triton server logs
	cd docker && $(DOCKER_COMPOSE) logs -f $(TRITON_SERVICE)

.PHONY: mlflow-up
mlflow-up: ## Start MLflow tracking server
	@echo "Starting MLflow tracking server..."
	cd docker && $(DOCKER_COMPOSE) up -d $(MLFLOW_SERVICE)
	@echo "✓ MLflow server starting at http://localhost:5000"

.PHONY: mlflow-down
mlflow-down: ## Stop MLflow tracking server
	@echo "Stopping MLflow tracking server..."
	cd docker && $(DOCKER_COMPOSE) down $(MLFLOW_SERVICE)
	@echo "✓ MLflow server stopped"

.PHONY: mlflow-logs
mlflow-logs: ## Show MLflow server logs
	cd docker && $(DOCKER_COMPOSE) logs -f $(MLFLOW_SERVICE)

.PHONY: services-up
services-up: mlflow-up triton-up ## Start all services (MLflow + Triton)
	@echo "✓ All services started"

.PHONY: services-down
services-down: triton-down mlflow-down ## Stop all services
	@echo "✓ All services stopped"

# API serving
.PHONY: api-dev
api-dev: ## Start FastAPI development server
	@echo "Starting FastAPI development server..."
	$(UVICORN) fedsense.serve_fastapi:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

.PHONY: api-prod
api-prod: ## Start FastAPI production server
	@echo "Starting FastAPI production server..."
	$(UVICORN) fedsense.serve_fastapi:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 4 \
		--log-level info

# Evaluation and results
.PHONY: evaluate
evaluate: ## Evaluate trained models
	@echo "Evaluating models..."
	mkdir -p $(RESULTS_DIR)
	@if [ -f "$(MODELS_DIR)/local_baseline/model.pkl" ]; then \
		echo "Evaluating local baseline..."; \
		$(PYTHON) -c "
from fedsense.eval import evaluate_model_on_dataset
import pickle

with open('$(MODELS_DIR)/local_baseline/model.pkl', 'rb') as f:
    state = pickle.load(f)

results = evaluate_model_on_dataset(
    state, 
    '$(DATA_DIR)/wearable_data.parquet',
    batch_size=64,
    window_length=250
)
print('Local Baseline Results:', results)
		"; \
	else \
		echo "No local baseline model found. Run: make train-local"; \
	fi

.PHONY: benchmark
benchmark: ## Run comprehensive benchmark
	@echo "Running comprehensive benchmark..."
	mkdir -p $(RESULTS_DIR)
	$(PYTHON) -c "
import time
import json
import os
from fedsense.datasets import generate_synthetic_wearable_data

# Generate test data for benchmarking
print('Generating benchmark data...')
start = time.time()
df = generate_synthetic_wearable_data(n_samples=10000, random_seed=42)
data_gen_time = time.time() - start

results = {
    'data_generation_time': data_gen_time,
    'data_samples': len(df),
    'anomaly_rate': df['label'].mean(),
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open('$(RESULTS_DIR)/benchmark.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Benchmark complete: {data_gen_time:.2f}s for {len(df)} samples')
	"

# Cleanup
.PHONY: clean-models
clean-models: ## Remove trained models
	rm -rf $(MODELS_DIR)
	@echo "✓ Trained models removed"

.PHONY: clean-logs
clean-logs: ## Remove log files
	rm -rf $(LOGS_DIR)
	@echo "✓ Log files removed"

.PHONY: clean-results
clean-results: ## Remove result files
	rm -rf $(RESULTS_DIR)
	@echo "✓ Result files removed"

.PHONY: clean-docker
clean-docker: ## Clean up Docker containers and volumes
	@echo "Cleaning up Docker resources..."
	cd docker && $(DOCKER_COMPOSE) down --volumes --remove-orphans
	docker system prune -f
	@echo "✓ Docker cleanup complete"

.PHONY: clean-all
clean-all: clean-venv clean-data clean-models clean-logs clean-results clean-docker ## Remove all generated files
	@echo "✓ Full cleanup complete"

# Development workflows
.PHONY: dev-setup
dev-setup: venv data ## Full development setup
	@echo "Development environment ready!"
	@echo "Next steps:"
	@echo "  1. Activate venv: source $(VENV_NAME)/bin/activate"
	@echo "  2. Train baseline: make train-local"
	@echo "  3. Start services: make services-up"
	@echo "  4. Run API: make api-dev"

.PHONY: demo
demo: data train-local services-up ## Run full demo pipeline
	@echo "Demo pipeline complete!"
	@echo "Services running:"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Triton: http://localhost:8001"
	@echo "Next: make api-dev"

.PHONY: ci
ci: check test ## Run CI pipeline (lint, type-check, test)
	@echo "✓ CI pipeline passed"

# Status and info
.PHONY: status
status: ## Show project status
	@echo "FedSense Project Status"
	@echo "======================"
	@echo "Environment:"
	@if [ -d "$(VENV_NAME)" ]; then echo "  ✓ Virtual environment ready"; else echo "  ✗ Virtual environment missing (run: make venv)"; fi
	@echo "Data:"
	@if [ -f "$(DATA_DIR)/wearable_data.parquet" ]; then echo "  ✓ Main dataset available"; else echo "  ✗ Main dataset missing (run: make data)"; fi
	@for i in 0 1 2 3 4; do \
		if [ -f "$(DATA_DIR)/client_$$i\_data.parquet" ]; then \
			echo "  ✓ Client $$i dataset available"; \
		else \
			echo "  ✗ Client $$i dataset missing"; \
		fi; \
	done
	@echo "Models:"
	@if [ -f "$(MODELS_DIR)/local_baseline/model.pkl" ]; then echo "  ✓ Local baseline trained"; else echo "  ✗ Local baseline missing (run: make train-local)"; fi
	@if [ -f "$(MODELS_DIR)/onnx/fedsense_model.onnx" ]; then echo "  ✓ ONNX model exported"; else echo "  ✗ ONNX model missing (run: make export-onnx)"; fi
	@echo "Services:"
	@if docker compose -f docker/docker-compose.yml ps $(MLFLOW_SERVICE) | grep -q "Up"; then echo "  ✓ MLflow running"; else echo "  ✗ MLflow stopped"; fi
	@if docker compose -f docker/docker-compose.yml ps $(TRITON_SERVICE) | grep -q "Up"; then echo "  ✓ Triton running"; else echo "  ✗ Triton stopped"; fi

.PHONY: docs
docs: ## Generate project documentation
	@echo "Generating documentation..."
	@echo "Project structure:" > README_GENERATED.md
	@echo '```' >> README_GENERATED.md
	@find . -type f -name "*.py" | head -20 | xargs -I {} echo {} >> README_GENERATED.md
	@echo '```' >> README_GENERATED.md
	@echo "Documentation generated in README_GENERATED.md"

# Make all targets phony by default
.DEFAULT_GOAL := help
