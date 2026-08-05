# Chameleon artifact -- run `make help` for the target list.
#
# Each target carries its own one-line description after `##`; `help` reads them
# from this file, so a target can never be missing from the help text without
# being visibly undocumented here.

SHELL         := /bin/bash
PYTHON        ?= python3
PROCS         ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
export LEVER_ROOT      := $(CURDIR)
export CHAM_PAPER_DIR  ?= output
export PYTHONPATH      := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))


GEN  := data_generator
OUT  := $(CHAM_PAPER_DIR)

.PHONY: help setup smoke test values data all experiments quick docker docker-multi clean distclean

help:  ## show this list
	@echo "Chameleon artifact"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	 | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  make %-14s %s\n", $$1, $$2}'

setup:  ## create ./venv and install pinned dependencies
	bash setup.sh

$(OUT):
	@mkdir -p $(OUT)/data

smoke: | $(OUT)  ## kick the tires (~2 min)
	@bash scripts/smoke.sh

test:  ## run the pytest suite
	@$(PYTHON) -m pytest tests/ -q

values: | $(OUT)  ## recompute the paper's reported numbers -> CSV
	@bash scripts/reproduce_results.sh values

data: | $(OUT)  ## export every measured result as CSV + XLSX
	@bash scripts/reproduce_results.sh data

all: | $(OUT)  ## values + data  (~15 s)
	@bash scripts/reproduce_results.sh all

experiments:  ## full re-measurement from scratch (core-days)
	@PROCS=$(PROCS) bash scripts/reproduce_all.sh

quick:  ## reduced-budget re-measurement, to check the pipeline runs
	@MODE=quick PROCS=$(PROCS) bash scripts/reproduce_all.sh

docker:  ## build the container image for this machine
	@bash docker/build.sh

docker-multi:  ## build the image for amd64 and arm64
	@bash docker/build.sh --multi

clean:  ## remove generated output and bytecode caches
	@rm -rf $(OUT)
	@find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@find . -name '*.pyc' -delete 2>/dev/null || true
	@echo "removed generated output and bytecode caches"

distclean: clean  ## clean, plus the virtualenv and build artifacts
	@rm -rf venv *.egg-info build dist
	@echo "removed the virtualenv and build artifacts"
