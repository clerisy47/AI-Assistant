.PHONY: up up-local down logs test eval ingest fmt-check build mlflow-experiment evidently-regression

up:            ## Start app + Qdrant, using a cloud LLM provider
	docker compose up --build

up-local:      ## Start app + Qdrant + a local vLLM server (needs a GPU)
	docker compose --profile local-llm up --build

down:          ## Stop and remove all containers
	docker compose down

logs:          ## Tail the app container's logs
	docker compose logs -f app

build:         ## Build the app image without starting anything
	docker compose build

ingest:        ## Ingest sample_docs/ into the running vector store
	docker compose exec app python scripts/ingest_sample_docs.py sample_docs

test:          ## Run the unit test suite locally (no Docker needed)
	uv sync --extra dev
	uv run pytest -v

eval:          ## Run verified-research eval harness (scripted; writes eval/report.md)
	uv sync --extra dev
	uv run python -m eval.harness

mlflow-experiment:  ## Phase 11: prompt matrix → MLflow + mlops/reports/mlflow_comparison.md
	uv sync --extra mlops --extra dev
	MLFLOW_DISABLE_AGENT_HINT=1 uv run --extra mlops python -m mlops.experiment_runner

evidently-regression:  ## Phase 12: golden set → Evidently suite + pct_tests_passed → MLflow
	uv sync --extra mlops --extra dev
	MLFLOW_DISABLE_AGENT_HINT=1 uv run --extra mlops python -m mlops.evidently_regression --prompt-version prompt_v3
