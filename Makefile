.PHONY: setup test lint format demo eval dashboard synth bench clean

UV ?= uv
export LLM_PROVIDER ?= fake

setup:
	$(UV) sync --all-extras

test:
	$(UV) run pytest

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

# Builds the golden set, runs baseline and a regressed candidate, calibrates the
# judge, compares, and gates. Exits 0 only when the gate correctly FAILS the
# regressed candidate.
demo:
	$(UV) run python scripts/demo.py

eval: demo

synth:
	$(UV) run python scripts/synth_traces.py --out traces/synthetic --report reports/before_after.md

bench:
	$(UV) run python scripts/bench_overhead.py --report reports/overhead.md

dashboard: synth
	$(UV) run llmops dashboard --traces traces/synthetic/baseline --compare traces/synthetic

clean:
	rm -rf reports traces/synthetic alerts .pytest_cache .ruff_cache
