# Common developer commands. Run `make help` to list them.
.DEFAULT_GOAL := help
.PHONY: help install install-dev lint format test check train dpo eval samples clean

help:           ## list the available commands
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:        ## GPU runtime stack + this package (editable)
	pip install -r requirements.txt

install-dev:    ## lint/test tooling + this package (CPU torch is enough)
	pip install torch -r requirements-dev.txt -e .

lint:           ## ruff lint + format check (what CI runs)
	ruff check .
	ruff format --check .

format:         ## auto-fix lint findings and reformat
	ruff check --fix .
	ruff format .

test:           ## run the unit tests (CPU, no downloads)
	pytest

check: lint test ## everything CI runs

train:          ## stage 1: SFT with the full-run settings from configs/full.env
	set -a; . configs/full.env; set +a; python scripts/train_sft.py

dpo:            ## stage 2: DPO on top of the SFT model
	python scripts/train_dpo.py

eval:           ## stage 3: lm-evaluation-harness benchmarks
	bash scripts/evaluate.sh

samples:        ## side-by-side SFT vs DPO answers
	python scripts/generate_samples.py

clean:          ## remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
