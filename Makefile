.PHONY: install install-dev lint format test check train dpo eval samples clean

install:        ## runtime stack for the GPU scripts
	pip install -r requirements.txt

install-dev:    ## lint/test tooling (CPU torch is enough)
	pip install torch -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff check --fix .

test:
	pytest

check: lint test

train:          ## SFT, full run settings from configs/full.env
	set -a; . configs/full.env; set +a; python train.py

dpo:
	python dpo.py

eval:
	bash eval.sh

samples:
	python samples.py

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.egg-info
