# --- Config -----------------------------------------------------------------

PYTHON        := python3
DBT           := dbt
PROJECT_ROOT  := $(PWD)
DBT_PROJECT   := llm_evals_dbt
DBT_PROFILES  := $(PROJECT_ROOT)      # profiles.yml lives in repo root
WAREHOUSE_DB  := warehouse/llm_evals.duckdb

# Entrypoint for your reasoning pipeline
PIPELINE_MODULE := src.core.pipeline.runs.reasoning

# --- Meta -------------------------------------------------------------------

.PHONY: help run smoke dbt-run dbt-test dbt-build test lint format \
        warehouse-init clean clean-db docker-build docker-run

help:
	@echo "LLM Evals Lab – common commands"
	@echo ""
	@echo "make run           - run full reasoning pipeline + dbt run + dbt test"
	@echo "make smoke         - tiny run (few samples) for quick sanity check"
	@echo "make dbt-run       - run dbt models"
	@echo "make dbt-test      - run dbt tests"
	@echo "make dbt-build     - dbt run + dbt test"
	@echo "make test          - run pytest suite"
	@echo "make clean-db      - remove DuckDB warehouse"
	@echo "make docker-build  - build Docker image"
	@echo "make docker-run    - run Docker image with local volumes mounted"

# --- Python pipeline --------------------------------------------------------

run:
	$(PYTHON) -m $(PIPELINE_MODULE)
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) run
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) test

# Small run – fewer samples, useful during development
smoke:
	$(PYTHON) -m $(PIPELINE_MODULE) --samples 5 || true
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) run
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) test

# --- dbt only ---------------------------------------------------------------

dbt-run:
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) run

dbt-test:
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) test

dbt-build:
	cd $(DBT_PROJECT) && DBT_PROFILES_DIR=$(DBT_PROFILES) $(DBT) build

# --- Python tests / quality -------------------------------------------------

test:
	$(PYTHON) -m pytest

# lint:
# 	$(PYTHON) -m ruff check src tests
# 
# format:
# 	$(PYTHON) -m ruff format src tests || true

# --- Warehouse management ---------------------------------------------------
warehouse-init:
	@mkdir -p warehouse data/raw
	@echo "Initialized warehouse + data dirs."

clean-db:
	rm -f $(WAREHOUSE_DB)

clean:
	rm -rf .pytest_cache .mypy_cache
	rm -rf $(DBT_PROJECT)/target
	rm -rf $(DBT_PROJECT)/dbt_packages

# --- Docker -----------------------------------------------------------------
# docker-build:
# 	docker build -t llm-evals-lab .
# 
# docker-run:
# 	docker run --rm \
# 	  --env-file .env \
# 	  -v $(PROJECT_ROOT)/warehouse:/app/warehouse \
# 	  -v $(PROJECT_ROOT)/data:/app/data \
# 	  llm-evals-lab
