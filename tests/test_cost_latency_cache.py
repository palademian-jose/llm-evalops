from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmops.cache import (
    CachedClient,
    ExactCache,
    HashedBoWEmbedder,
    MemoryLRU,
    SemanticCache,
    cache_stats,
    overall_hit_rate,
)
from llmops.cli import app
from llmops.cost import cost_table, read_bill, reconcile
from llmops.latency import latency_table
from llmops.llm import FakeClient, FakeConfig, Message
from llmops.pricing import estimate_cost
from llmops.tracing import JsonlSink, MemorySink, Span, Tracer

runner = CliRunner()


def _span(**kw) -> Span:
    base = dict(
        name="generate",
        kind="llm",
        start_ts="2026-10-01T10:00:00.000Z",
        end_ts="2026-10-01T10:00:01.000Z",
    )
    base.update(kw)
    return Span(**base)


def _spans() -> list[Span]:
    spans = []
    for i in range(4):
        tid = f"t{i}"
        spans.append(
            _span(
                trace_id=tid,
                span_id=f"root{i}",
                kind="other",
                name="rag",
                feature="rag.answer",
                user_id=f"u{i % 2}",
                latency_ms=1000 + i * 100,
                model=None,
            )
        )
        spans.append(
            _span(
                trace_id=tid,
                span_id=f"r{i}",
                parent_span_id=f"root{i}",
                kind="retrieval",
                name="retrieve",
                feature="rag.answer",
                user_id=f"u{i % 2}",
                latency_ms=200,
                model=None,
            )
        )
        spans.append(
            _span(
                trace_id=tid,
                span_id=f"l{i}",
                parent_span_id=f"root{i}",
                feature="rag.answer",
                user_id=f"u{i % 2}",
                latency_ms=700,
                model="anthropic/claude-sonnet-4.5",
                input_tokens=1000,
                output_tokens=500,
                cost_usd=estimate_cost("anthropic/claude-sonnet-4.5", 1000, 500),
                meta={"ttft_ms": 120 + i},
            )
        )
    spans.append(
        _span(
            trace_id="t9",
            span_id="root9",
            kind="other",
            name="agent",
            feature="agent.run",
            start_ts="2026-10-02T10:00:00.000Z",
            latency_ms=3000,
            model=None,
        )
    )
    spans.append(
        _span(
            trace_id="t9",
            span_id="l9",
            parent_span_id="root9",
            feature="agent.run",
            start_ts="2026-10-02T10:00:00.000Z",
            latency_ms=2500,
            model="openai/gpt-4.1-mini",
            input_tokens=2000,
            output_tokens=100,
            cost_usd=estimate_cost("openai/gpt-4.1-mini", 2000, 100),
        )
    )
    return spans


def test_cost_table_groupings_match_pricing():
    spans = _spans()
    by_model = {r.key: r for r in cost_table(spans, "model")}
    assert by_model["anthropic/claude-sonnet-4.5"].cost_usd == pytest.approx(4 * (0.003 + 0.0075))
    assert by_model["anthropic/claude-sonnet-4.5"].requests == 4
    assert by_model["openai/gpt-4.1-mini"].cost_usd == pytest.approx(0.0008 + 0.00016)
    by_feature = cost_table(spans, "feature")
    assert by_feature[0].key == "rag.answer" and by_feature[0].requests == 4
    assert {r.key for r in cost_table(spans, "user")} == {"u0", "u1", "anon"}
    days = cost_table(spans, "day")
    assert [r.key for r in days] == ["2026-10-01", "2026-10-02"]
    assert len(cost_table(spans, "request")) == 5
    assert by_feature[0].cost_per_1k_requests == pytest.approx(by_feature[0].cost_usd / 4 * 1000)


def test_reconcile_against_simple_and_openrouter_csv(tmp_path: Path):
    spans = _spans()
    bill = tmp_path / "bill.csv"
    bill.write_text(
        "date,model,input_tokens,output_tokens,cost_usd\n"
        "2026-10-01,claude-sonnet-4-5,4000,2000,0.0420\n"
        "2026-10-02,openai/gpt-4.1-mini,2100,0,0.0010\n",
        encoding="utf-8",
    )
    rep = reconcile(spans, bill)
    assert rep.bill_format == "simple" and len(rep.rows) == 2
    sonnet = rep.rows[0]
    assert sonnet.model == "anthropic/claude-sonnet-4.5" and sonnet.computed_usd == pytest.approx(
        0.042
    )
    assert sonnet.diff_pct == pytest.approx(0.0, abs=1e-6)
    assert rep.discrepancy_pct == pytest.approx((0.042 + 0.00096 - 0.043) / 0.043 * 100)
    assert "discrepancy" in rep.to_markdown()

    orx = tmp_path / "orx.csv"
    orx.write_text(
        "created_at,model,app,tokens_prompt,tokens_completion,cost_total\n"
        "2026-10-01T09:00:00Z,anthropic/claude-sonnet-4.5,x,2000,1000,0.021\n"
        "2026-10-01T10:00:00Z,anthropic/claude-sonnet-4.5,x,2000,1000,0.021\n",
        encoding="utf-8",
    )
    lines, fmt = read_bill(orx)
    assert fmt == "openrouter" and lines[0].day == "2026-10-01" and lines[0].cost_usd == 0.021
    rep2 = reconcile(spans, orx)
    row = next(r for r in rep2.rows if r.model == "anthropic/claude-sonnet-4.5")
    assert row.billed_usd == pytest.approx(0.042) and row.billed_tokens == 6000
    with pytest.raises(ValueError):
        bad = tmp_path / "bad.csv"
        bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
        read_bill(bad)


def test_latency_percentiles_and_stage_breakdown():
    spans = _spans()
    by_feature = {r.key: r for r in latency_table(spans, "feature")}
    assert by_feature["rag.answer"].n == 4 and by_feature["rag.answer"].p50 == pytest.approx(1150)
    assert by_feature["rag.answer"].p99 == pytest.approx(1297)
    assert by_feature["rag.answer"].ttft_p50 == pytest.approx(121.5)
    assert by_feature["agent.run"].ttft_p50 is None
    stages = {r.key: r for r in latency_table(spans, "stage")}
    assert stages["llm"].n == 5 and stages["retrieval"].p95 == 200
    total_root = 1000 + 1100 + 1200 + 1300 + 3000
    assert stages["llm"].share == pytest.approx((4 * 700 + 2500) / total_root)
    assert stages["retrieval"].share == pytest.approx(800 / total_root)


def test_cli_cost_latency_cache_stats(tmp_path: Path):
    path = tmp_path / "spans.jsonl"
    sink = JsonlSink(path)
    for s in _spans():
        sink.write(s)
    res = runner.invoke(app, ["cost", "--traces", str(path), "--by", "model"])
    assert res.exit_code == 0, res.output
    assert "anthropic/claude-sonnet-4.5" in res.output and "$/1k req" in res.output
    res = runner.invoke(
        app,
        [
            "cost",
            "--traces",
            str(tmp_path),
            "--by",
            "day",
            "--validate-against",
            "examples/bill_sample.csv",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "discrepancy" in res.output
    res = runner.invoke(app, ["latency", "--traces", str(path), "--by", "stage"])
    assert res.exit_code == 0 and "share of request time" in res.output
    res = runner.invoke(app, ["cache-stats", "--traces", str(path)])
    assert res.exit_code == 0 and "no cache spans" in res.output


def test_exact_cache_hit_miss_and_saved_amounts():
    sink = MemorySink()
    tracer = Tracer(sink)
    exact = ExactCache(MemoryLRU(maxsize=2), tracer=tracer)
    client = CachedClient(
        FakeClient(FakeConfig(simulated_latency_ms=50)), exact=exact, tracer=tracer
    )
    msgs = [Message(role="user", content="hello there")]
    r1 = client.complete(msgs, model="openai/gpt-4o")
    r2 = client.complete(msgs, model="openai/gpt-4o")
    assert r1.text == r2.text and exact.stats()["hits"] == 1 and exact.stats()["misses"] == 1
    client.complete(msgs, model="openai/gpt-4o", temperature=0.7)  # different params -> miss
    assert exact.stats()["misses"] == 2
    cache_spans = [s for s in sink.spans if s.kind == "cache"]
    assert [s.cache_hit for s in cache_spans] == [False, True, False]
    hit = cache_spans[1]
    expected_cost = estimate_cost("openai/gpt-4o", r1.usage.input_tokens, r1.usage.output_tokens)
    assert hit.meta["saved_cost_usd"] == pytest.approx(expected_cost)
    assert hit.meta["saved_ms"] == pytest.approx(r1.latency_ms)
    assert sum(1 for s in sink.spans if s.kind == "llm") == 2
    # LRU eviction
    client.complete([Message(role="user", content="third")], model="openai/gpt-4o")
    assert len(exact.backend) == 2
    rows = cache_stats(sink.spans)
    assert (
        rows[0].cache_type == "exact"
        and rows[0].hits == 1
        and rows[0].saved_cost_usd == pytest.approx(expected_cost)
    )
    assert 0 < overall_hit_rate(sink.spans) < 1


def test_semantic_cache_threshold_controls_hits():
    emb = HashedBoWEmbedder(dim=256)
    a = "What is the capital city of France?"
    b = "What is the capital city of France ?"
    c = "How do I bake sourdough bread at home"
    va, vb, vc = emb.embed([a, b, c])
    sim_ab, sim_ac = float(va @ vb), float(va @ vc)
    assert sim_ab > 0.9 > sim_ac

    sink = MemorySink()
    tracer = Tracer(sink)
    sem = SemanticCache(emb, threshold=0.9, tracer=tracer)
    client = CachedClient(FakeClient(), semantic=sem, tracer=tracer)
    client.complete([Message(role="user", content=a)], model="m")
    assert client.complete([Message(role="user", content=b)], model="m").text == "FAKE: " + a
    assert client.complete([Message(role="user", content=c)], model="m").text == "FAKE: " + c
    assert sem.stats() == {"hits": 1, "misses": 2, "hit_rate": 1 / 3, "size": 2}
    hit_span = next(s for s in sink.spans if s.kind == "cache" and s.cache_hit)
    assert hit_span.meta["similarity"] >= 0.9 and hit_span.meta["cache_type"] == "semantic"
    # Different scope (model) never matches.
    assert client.complete([Message(role="user", content=b)], model="other").text == "FAKE: " + b
    near = "What is the capital of France?"  # drops one token: similar but not identical
    assert 0.5 < float(va @ emb.embed([near])[0]) < 0.999
    strict = SemanticCache(emb, threshold=0.999, tracer=tracer)
    strict_client = CachedClient(FakeClient(), semantic=strict, tracer=tracer)
    strict_client.complete([Message(role="user", content=a)], model="m")
    strict_client.complete([Message(role="user", content=near)], model="m")
    assert strict.stats()["hits"] == 0
    loose = SemanticCache(emb, threshold=0.5, tracer=tracer)
    loose_client = CachedClient(FakeClient(), semantic=loose, tracer=tracer)
    loose_client.complete([Message(role="user", content=a)], model="m")
    loose_client.complete([Message(role="user", content=near)], model="m")
    assert loose.stats()["hits"] == 1


def test_redis_backend_with_fake_client():
    class FakeRedis:
        def __init__(self):
            self.d = {}

        def get(self, k):
            return self.d.get(k)

        def set(self, k, v, ex=None):
            self.d[k] = v

        def scan_iter(self, pattern):
            return iter([k for k in self.d if k.startswith(pattern[:-1])])

    from llmops.cache import RedisBackend

    b = RedisBackend("redis://unused", client=FakeRedis())
    b.set("k", {"x": 1}, ttl=10)
    assert b.get("k") == {"x": 1} and b.get("nope") is None and len(b) == 1
