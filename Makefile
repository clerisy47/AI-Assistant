.PHONY: up up-local down logs test ingest fmt-check build

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
	pip install -r requirements-dev.txt --break-system-packages
	pytest -v
