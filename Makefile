# Ingestion pipeline, end to end. Each stage gates the next: a failing stage
# returns a non-zero exit code and make stops, which is the whole point of
# having gates rather than warnings.

PY := uv run python
COUNT ?= 50000
SEED ?= 42
FAIL_UNDER ?= 0.85

.PHONY: pipeline generate validate embed publish probe test test-slow clean help

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

pipeline: generate validate embed publish probe ## Run every stage from raw data to a published bundle

generate: ## Synthesise raw ads with seeded defects
	$(PY) scripts/generate_raw_ads.py --count $(COUNT) --seed $(SEED)

validate: ## Phase 1: clean + reject, gated on pass rate
	$(PY) -m ingestion.validate --fail-under $(FAIL_UNDER)
	$(PY) scripts/check_oracle.py

embed: ## Phase 2: generate embeddings, gated on the token limit
	$(PY) -m ingestion.embed

publish: ## Phase 2: verify and swap a versioned bundle into place
	$(PY) -m ingestion.publish

probe: ## Sanity-check the published vectors with brute-force search
	$(PY) scripts/probe_embeddings.py

test: ## Fast suite (no model download)
	uv run pytest

test-slow: ## Tests that load the real model
	uv run pytest -m slow

clean: ## Delete all generated data
	rm -rf data/raw/* data/clean/* data/embedded data/published
