# Event parser — common dev commands.
# Run `make help` for a summary.

VENV       := .venv
PY         := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
PYTEST     := $(VENV)/bin/pytest
HOST       ?= 127.0.0.1
PORT       ?= 5001

.DEFAULT_GOAL := help

.PHONY: help install up run test demo reset clean distclean env

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

$(VENV)/bin/activate: requirements.txt ## (internal) Create venv + install deps
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install --quiet --upgrade pip
	@$(PIP) install --quiet -r requirements.txt
	@touch $(VENV)/bin/activate

install: $(VENV)/bin/activate ## Create venv and install dependencies

env: ## Copy .env.example to .env if missing
	@test -f .env || cp .env.example .env
	@echo ".env ready"

up: install env ## Start the Flask app + UI at http://$(HOST):$(PORT)
	@echo "Event Parser UI → http://$(HOST):$(PORT)"
	@FLASK_HOST=$(HOST) FLASK_PORT=$(PORT) $(PY) app.py

run: up ## Alias for `up`

test: install ## Run the test suite
	@$(PYTEST) -q

demo: install ## Run the cold→warm batch showcase in the terminal
	@$(PY) scripts/demo.py

ROWS ?= 10000
SEED ?= 42
OUT  ?= sample_logs/chaos.log

logs: install ## Generate synthetic chaotic logs (override: ROWS=50000 OUT=... SEED=...)
	@$(PY) scripts/generate_logs.py --rows $(ROWS) --seed $(SEED) --out $(OUT)

samples: install ## (Re)generate train/test sample logs for the demo (8 datasets × 2 splits = 16 files)
	@$(PY) scripts/generate_samples.py

reset: ## Clear all per-customer storage (parsers, events, quarantine, clusters)
	@rm -rf storage/parsers/*.jsonl storage/events/*.jsonl storage/quarantine/*.jsonl storage/clusters/*.jsonl 2>/dev/null || true
	@echo "storage/ cleared"

clean: ## Remove caches and storage (keeps .venv)
	@rm -rf __pycache__ .pytest_cache **/__pycache__ storage/**/*.jsonl
	@echo "cleaned"

distclean: clean ## Remove caches, storage, and venv
	@rm -rf $(VENV)
	@echo "venv removed"
