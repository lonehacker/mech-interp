.PHONY: help test lint lint-fix format check audit

help:
	@echo "Targets:"
	@echo "  test       — run unit tests (53 tests, no model load)"
	@echo "  lint       — ruff check src/ experiments/ tests/"
	@echo "  lint-fix   — ruff check --fix (auto-fix safe rules only)"
	@echo "  format     — ruff format src/ experiments/ tests/"
	@echo "  check      — lint + test (pre-commit smoke)"
	@echo "  audit      — re-run vocab + length audit on data/code_contrastive_matched.jsonl"

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ experiments/ tests/

lint-fix:
	ruff check src/ experiments/ tests/ --fix

format:
	ruff format src/ experiments/ tests/

check: lint test

audit:
	python experiments/matched_dual_audit.py
	python experiments/matched_shuffle_control.py
