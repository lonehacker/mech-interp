# Convenience targets. Run from the project root.
#
# First-time setup:
#   make setup       # create .venv, install pinned deps + package in editable mode
#
# Common loops:
#   make test        # pytest (no model load, ~5s)
#   make verify      # reproducibility harness only (cache-key + extract_d_hat invariants)
#   make lint        # ruff check
#   make check       # lint + test (pre-commit smoke)
#   make clean       # remove __pycache__, .pytest_cache, .ruff_cache, *.egg-info
#
# Notes:
#   - All targets assume the venv at .venv/ is active OR Python is on PATH.
#     The `setup` target creates .venv/ — activate with `source .venv/bin/activate`.
#   - `make clean` does NOT touch artifacts/cache/ (the activation cache is
#     load-bearing for Phase 1/2 reproducibility — see PROJECT_STATE.md).

PY := python
PKG_DIRS := mech_security experiments tests

.PHONY: help setup test verify lint lint-fix format check audit clean

help:
	@echo "Targets:"
	@echo "  setup      - create .venv and install pinned deps + editable package"
	@echo "  test       - run all unit tests (no model load)"
	@echo "  verify     - reproducibility harness only (cache-key + numerical invariants)"
	@echo "  lint       - ruff check $(PKG_DIRS)"
	@echo "  lint-fix   - ruff check --fix (auto-fix safe rules)"
	@echo "  format     - ruff format $(PKG_DIRS)"
	@echo "  check      - lint + test (pre-commit smoke)"
	@echo "  audit      - re-run vocab + length audit on data/code_contrastive_matched.jsonl"
	@echo "  clean      - remove __pycache__, .pytest_cache, .ruff_cache, *.egg-info"

setup:
	@if [ ! -d .venv ]; then \
		echo "Creating .venv with Python 3.11..." ; \
		$(PY) -m venv .venv ; \
	fi
	@echo "Installing pinned deps + editable package..."
	@.venv/bin/pip install --upgrade pip
	@.venv/bin/pip install -r requirements.txt
	@.venv/bin/pip install -e .
	@echo ""
	@echo "Done. Activate with: source .venv/bin/activate"
	@echo "Then: make test"

test:
	$(PY) -m pytest tests/ -v

verify:
	$(PY) -m pytest tests/test_reproducibility.py -v

lint:
	ruff check $(PKG_DIRS)

lint-fix:
	ruff check $(PKG_DIRS) --fix

format:
	ruff format $(PKG_DIRS)

check: lint test

audit:
	$(PY) -m experiments.matched_dual_audit
	$(PY) -m experiments.matched_shuffle_control

clean:
	@echo "Removing __pycache__ directories..."
	@find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Removing .pytest_cache, .ruff_cache, *.egg-info..."
	@rm -rf .pytest_cache .ruff_cache mech_security.egg-info
	@echo "Done. artifacts/cache/ left alone (load-bearing — see PROJECT_STATE.md)."
