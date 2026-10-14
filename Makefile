.PHONY: setup test lint format demo eval dashboard synth real-before-after bench clean

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

# The same before/after table from the REAL docs-rag scenario traces (fake provider, projected
# cost). Run the scenarios first: `cd ../docs-rag && make traces`.
DOCS_RAG ?= ../docs-rag
real-before-after:
	$(UV) run llmops before-after --dirs "$(DOCS_RAG)/traces/scenarios/*" --md reports/before_after_docs_rag.md --title "docs-rag serving scenarios (fake provider, projected cost)" --note "Fake provider: real docs-rag pipeline runs over one shared request stream; the LLM has no real latency or bill, so cost per 1k requests is projected from tokens at list price and p50/p95 are pipeline compute only."

bench:
	$(UV) run python scripts/bench_overhead.py --report reports/overhead.md

dashboard: synth
	$(UV) run llmops dashboard --traces traces/synthetic/baseline --compare traces/synthetic

clean:
	rm -rf reports traces/synthetic alerts .pytest_cache .ruff_cache
