# Integrating `llm-evalops` into `docs-rag` and `briefing-agent`

`llm-evalops` is one distribution with two packages: `evalkit` (golden sets, judge, statistics,
gate) and `llmops` (tracing, cost, latency, cache, alerts, dashboard). It is the source of truth
for the trace span schema, `pricing.json`, the golden/eval-result formats, the gate CLI and the
judge interface defined in `PLAN.md`.

## 1. Add the dependency

Locally, while the three repos live side by side:

```bash
cd ../docs-rag           # or ../briefing-agent
uv add ../llm-evalops    # path dependency; uv records it in pyproject.toml
uv sync --all-extras
```

Later, from git (no PyPI release needed):

```bash
uv add "llm-evalops @ git+https://github.com/<owner>/llm-evalops.git"
```

CI in the consumer: `actions/checkout` cannot write outside `$GITHUB_WORKSPACE`, so check the
consumer out to `<name>/` and this repo to `llm-evalops/` (`repository: <owner>/llm-evalops`,
`path: llm-evalops`), set `defaults.run.working-directory: <name>`, and `../llm-evalops`
resolves. This repo must be public, or pass `token:` with a read-only secret. Docker needs the
sibling as a named build context (`docker build --build-context llm-evalops=../llm-evalops .`)
and `uv sync --no-editable`.

Optional extras: `dashboard` (Streamlit), `otel` (OpenTelemetry export), `redis` (exact cache
backend).

Both consumers already ship an OpenAI-compatible client layer; nothing forces them to use
`llmops.llm`, but they may (`from llmops.llm import get_client, Message, ToolSpec`) and drop
their local copy. `get_client()` picks `openrouter` when `OPENROUTER_API_KEY` is set, then
`anthropic` / `openai` by key, else `fake`.

## 2. Swap the local `tracing.py` for `llmops.tracing`

The local 60-line writer and `llmops.tracing` produce the same JSONL schema, so trace files
written before the swap stay readable. Replace the module with re-exports:

```python
# src/docs_rag/tracing.py  (was a local JSONL writer)
from llmops.tracing import Tracer, get_tracer, span, trace, wrap_client  # noqa: F401
```

Then:

```python
from llmops.tracing import span, trace, wrap_client

client = wrap_client(get_client(), feature="rag.answer")  # every complete() -> one llm span


def answer(question: str, user_id: str = "anon") -> dict:
    with trace(feature="rag.answer", user_id=user_id):
        with span("retrieve", "retrieval", meta={"k": 20}) as s:
            docs = retriever.search(question, k=20)
            s.meta["hits"] = len(docs)
        with span("rerank", "rerank", model="rerank-v3.5") as s:
            docs = reranker.rerank(question, docs)
            s.cost_usd = estimate_cost("rerank-v3.5", searches=1) or 0.0
        resp = client.complete([...])
        return {
            "answer": resp.text,
            "retrieved": [d.id for d in docs],
            "contexts": [d.text for d in docs],
        }
```

Rules that keep the data useful:

- One `trace(...)` per user request; one root span per trace. Nested `span()` calls inherit
  `trace_id`, `feature`, `user_id`, `session_id`.
- Use the span kinds from the contract (`llm`, `retrieval`, `rerank`, `tool`, `cache`,
  `verify`, `other`); `llmops latency --by stage` and the dashboard group by them.
- Put `ttft_ms` into `meta` on llm spans when streaming, `refusal: bool` or `output_text`
  when you want refusal-rate alerts.
- `TRACE_PATH` points at the JSONL file (default `traces/spans.jsonl`). In multi-process
  servers use `JsonlSink(path, per_process=True)` and ingest the directory.
- Cost comes from `src/llmops/pricing.json`; copy it into the consumer repo only if the consumer
  must run without `llmops` installed, and treat `llmops` as the source of truth. When the
  provider reports cost (OpenRouter `usage.cost`), `wrap_client` prefers it and stores both.

## 3. Swap the local `gate.py` for `evalkit gate`

Delete the vendored gate and call the CLI in CI:

```yaml
- run: uv run evalkit run --golden evals/golden/rag.jsonl --target docs_rag.eval_target:answer \
         --metrics recall_at_20,faithfulness,answer_relevance,task_success --out reports --run-id pr
- run: uv run evalkit gate --baseline reports/eval-main.json --candidate reports/eval-pr.json \
         --config evals.yaml --md reports/gate.md --annotate
```

`evals.yaml` keeps the contract shape:

```yaml
min_cases: 30
metrics:
  faithfulness: {min: 0.85, max_drop: 0.05}
  recall_at_20: {min: 0.80, max_drop: 0.05}
```

The eval target is any `(input: dict) -> dict` callable. The field conventions the built-in
metrics read are documented in the README ("Metric definitions"):
`output.answer`, `output.retrieved` (ranked ids), `output.contexts`, `output.fields`,
`expected.answer`, `expected.relevant`, `expected.fields`. Agent trajectories can be returned
under any key and scored with a custom verifier (`--verifier pkg.module:callable`) or a custom
metric registered with `@evalkit.metrics.metric("name")` in the consumer package.

Or reuse the composite action from this repo (`ci/action.yml`) or the reusable workflow
(`ci/eval-gate.yml`), which downloads the `eval-baseline` artifact from `main`.

## 4. Judge

```python
from evalkit.judge import judge  # contract 4: judge(criterion, input, output, reference)

verdict = judge("faithfulness", {"question": q}, {"answer": a}, {"contexts": ctxs})
# {"pass": bool, "reason": str}
```

The judge model is `JUDGE_MODEL` (default `openai/gpt-4.1-mini`), which must be a different
family than `LLM_MODEL`; `evalkit calibrate --strict-family` enforces it. Calibrate against
`evals/human_labels.jsonl` before trusting a judge-backed metric in a gate.

## 5. Dashboard and before/after across repos

`llmops dashboard --traces ../docs-rag/traces ../briefing-agent/traces --compare traces/before-after`
reads any number of trace sources. For P1's caching work, write one directory of spans per
serving configuration (`docs-rag/scripts/trace_scenarios.py` does this: `baseline`, `exact_cache`,
`semantic_cache`, `cheaper_model` under `traces/scenarios/`) and build the $/1k-requests, p95 and
cache hit-rate table with

```bash
uv run llmops before-after --dirs '../docs-rag/traces/scenarios/*' --md reports/before_after_docs_rag.md
```

`--dirs` (and the dashboard's `--compare`) accept a scenario directory, a parent directory with
one sub-directory per scenario, a `.jsonl` file, or a quoted glob; the scenario called `baseline`
is the reference for the delta columns (`--baseline NAME` overrides). The same code backs
`scripts/synth_traces.py --from-dirs`. Roots must not repeat their children's cost: `llmops`
sums `cost_usd` over every span of a trace, so put rolled-up totals in `meta`, not in the root's
`cost_usd`.

## 6. Checklist

- [ ] `uv add ../llm-evalops`; `.env` has `OPENROUTER_API_KEY` or `LLM_PROVIDER=fake`.
- [ ] `tracing.py` re-exports `llmops.tracing`; `wrap_client` wraps the LLM client.
- [ ] `gate.py` removed; CI calls `evalkit run` + `evalkit gate`.
- [ ] `evals/golden/*.jsonl` validates with `evalkit validate --strict`.
- [ ] `evalkit calibrate` kappa >= 0.6 for every judge-backed metric used in `evals.yaml`.
- [ ] `alerts.yaml` copied and `llmops check` scheduled (see `ci/alerts-schedule.yml`).

## 7. Interface gaps found integrating `docs-rag` and `briefing-agent`

| Gap | Resolution |
|---|---|
| Consumers called the provider-billed cost `reported_cost_usd`; `wrap_client` reads `LLMResponse.cost_usd` | Consumers renamed the field to `cost_usd` (matches `llmops.llm.types`) |
| `wrap_client` guessed `gen_ai.system` from the class name (`OpenAICompatClient` -> `openai` even for OpenRouter) | Uses the client's `provider` attribute first; `system=` overrides |
| Agents name every call (`agent.act`, `agent.compose`) and add `run_id` / `phase` | `wrap_client(..., name=, meta=)`, one wrapper per call; `response_cost(resp)` exposes the reported-over-computed rule for budget accounting |
| Root spans repeated their children's `cost_usd`, so `llmops` double-counted | Rolled-up totals go in `meta.total_cost_usd`; only leaf spans carry cost |
| Consumers' price tables had extra ids (`fake-*`, `per_call` rerank price) | Fake ids added to `llmops/pricing.json` at zero cost; rerank is `per_search`; consumers ship no copy |
| Gate configs used `max_rise` and `optional: true` | `evals.yaml` uses `max_increase`; `optional` is now an `evalkit` threshold key (status `SKIP`) |
| Reports stored unmeasured metrics as `null` and per-case `runs` as a list of records | `EvalResult` reads `null` metrics as absent; `CaseResult.runs` accepts `int` or a list |
| Fake-provider judges differ (`evalkit` fake judge vs each consumer's lexical rubric) | Consumers keep their lexical judge for `LLM_PROVIDER=fake` (committed baselines depend on it) and use `evalkit.judge.Judge` for real providers |
