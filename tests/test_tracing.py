from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llmops.ingest import load_spans
from llmops.llm import FakeClient, FakeConfig, LLMResponse, Message, Usage
from llmops.pricing import estimate_cost
from llmops.tracing import JsonlSink, MemorySink, Span, Tracer, wrap_client


def test_nested_spans_share_trace_and_link_parents():
    sink = MemorySink()
    t = Tracer(sink, feature="rag.answer", user_id="u1")
    with t.trace(session_id="s1") as tid:
        with t.span("outer", "other") as outer:
            with t.span("retrieve", "retrieval", meta={"k": 5}) as r:
                pass
            with t.span("generate", "llm", model="openai/gpt-4o") as g:
                g.set(input_tokens=10, output_tokens=5, custom="x")
    kinds = {s.name: s for s in sink.spans}
    assert kinds["retrieve"].parent_span_id == outer.span_id
    assert kinds["generate"].parent_span_id == outer.span_id
    assert outer.parent_span_id is None
    assert {s.trace_id for s in sink.spans} == {tid}
    assert all(
        s.feature == "rag.answer" and s.user_id == "u1" and s.session_id == "s1" for s in sink.spans
    )
    assert kinds["retrieve"].meta == {"k": 5}
    assert kinds["generate"].meta["custom"] == "x" and kinds["generate"].input_tokens == 10
    assert all(s.end_ts and s.latency_ms >= 0 for s in sink.spans)
    assert r.kind == "retrieval"


def test_span_without_trace_starts_its_own_and_records_errors():
    sink = MemorySink()
    t = Tracer(sink)
    with pytest.raises(ValueError):
        with t.span("boom", "tool"):
            raise ValueError("bad")
    with t.span("ok"):
        pass
    boom, ok = sink.spans
    assert (
        boom.status == "error" and boom.error_kind == "ValueError" and boom.meta["error"] == "bad"
    )
    assert boom.trace_id != ok.trace_id and ok.status == "ok"


def test_span_json_matches_plan_schema(tmp_path: Path):
    sink = JsonlSink(tmp_path / "spans.jsonl")
    t = Tracer(sink)
    with t.span("generate", "llm", model="openai/gpt-4o"):
        pass
    line = json.loads((tmp_path / "spans.jsonl").read_text(encoding="utf-8").splitlines()[0])
    expected_keys = {
        "trace_id",
        "span_id",
        "parent_span_id",
        "name",
        "kind",
        "start_ts",
        "end_ts",
        "latency_ms",
        "model",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "cache_hit",
        "feature",
        "user_id",
        "session_id",
        "status",
        "error_kind",
        "meta",
    }
    assert expected_keys <= set(line)
    assert line["start_ts"].endswith("Z") and line["kind"] == "llm"


def test_jsonl_integrity_under_concurrent_threads(tmp_path: Path):
    path = tmp_path / "spans.jsonl"
    sink = JsonlSink(path)
    t = Tracer(sink)

    def work(i: int) -> None:
        with t.trace(feature=f"f{i % 3}"):
            for j in range(100):
                with t.span(f"s{j}", "tool", meta={"payload": "x" * 200}):
                    pass

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(8)))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 800
    assert all(json.loads(line)["name"].startswith("s") for line in lines)
    assert len({json.loads(line)["trace_id"] for line in lines}) == 8
    assert len(load_spans(path)) == 800


def test_jsonl_integrity_under_concurrent_processes(tmp_path: Path, env: dict[str, str]):
    path = tmp_path / "spans.jsonl"
    code = (
        "import sys\n"
        "from llmops.tracing import JsonlSink, Tracer\n"
        "t = Tracer(JsonlSink(sys.argv[1]))\n"
        "for i in range(150):\n"
        "    with t.span('p' + sys.argv[2], 'tool', meta={'pad': 'y' * 300}):\n"
        "        pass\n"
    )
    procs = [
        subprocess.Popen([sys.executable, "-c", code, str(path), str(i)], env=env) for i in range(3)
    ]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 450 and all(json.loads(line)["kind"] == "tool" for line in lines)


def test_wrap_client_records_model_tokens_latency_and_computed_cost():
    sink = MemorySink()
    t = Tracer(sink)
    client = wrap_client(FakeClient(FakeConfig(simulated_latency_ms=12.5)), t, feature="chat")
    resp = client.complete([Message(role="user", content="hello world")], model="openai/gpt-4o")
    (s,) = sink.spans
    assert s.kind == "llm" and s.model == "openai/gpt-4o" and s.feature == "chat"
    assert s.input_tokens == resp.usage.input_tokens and s.output_tokens == resp.usage.output_tokens
    assert s.latency_ms == pytest.approx(resp.latency_ms, abs=1e-3)
    assert s.cost_usd == estimate_cost("openai/gpt-4o", s.input_tokens, s.output_tokens)
    assert s.meta["cost_source"] == "computed" and s.meta["cost_reported_usd"] is None
    assert s.meta["gen_ai.system"] == "fake"
    assert wrap_client(client) is client


def test_wrap_client_prefers_provider_reported_cost():
    class Reporting:
        def complete(self, messages, *, tools=None, model=None, temperature=0.0, max_tokens=1024):
            return LLMResponse(
                text="x",
                usage=Usage(input_tokens=1000, output_tokens=1000),
                model="anthropic/claude-sonnet-4.5",
                latency_ms=1.0,
                cost_usd=0.02,
            )

    sink = MemorySink()
    client = wrap_client(Reporting(), Tracer(sink))
    client.complete([Message(role="user", content="hi")])
    (s,) = sink.spans
    assert s.cost_usd == 0.02
    assert s.meta["cost_source"] == "reported"
    assert s.meta["cost_computed_usd"] == pytest.approx(0.018)
    assert s.meta["cost_reported_usd"] == 0.02


def test_wrap_client_unknown_model_costs_zero_and_errors_propagate():
    class Broken:
        def complete(self, messages, **kw):
            raise RuntimeError("provider down")

    sink = MemorySink()
    with pytest.raises(RuntimeError):
        wrap_client(Broken(), Tracer(sink)).complete([Message(role="user", content="hi")])
    assert sink.spans[0].status == "error" and sink.spans[0].error_kind == "RuntimeError"

    client = wrap_client(FakeClient(FakeConfig(model="mystery-9000")), Tracer(sink))
    client.complete([Message(role="user", content="hi")])
    assert sink.spans[-1].cost_usd == 0.0 and sink.spans[-1].meta["cost_source"] == "unknown_model"


def test_otel_exporter_emits_genai_attributes():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from llmops.tracing.otel import OTelExporter

    provider = TracerProvider()
    mem = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(mem))
    t = Tracer(MemorySink(), exporters=[OTelExporter(provider)])
    with t.trace(feature="rag.answer"):
        with t.span("root", "other"):
            with t.span("generate", "llm", model="openai/gpt-4o") as s:
                s.set(input_tokens=7, output_tokens=3, meta={"stop_reason": "stop"})
    finished = {s.name: s for s in mem.get_finished_spans()}
    gen = finished["generate"]
    assert gen.attributes["gen_ai.system"] == "openai"
    assert gen.attributes["gen_ai.request.model"] == "openai/gpt-4o"
    assert gen.attributes["gen_ai.usage.input_tokens"] == 7
    assert gen.attributes["gen_ai.usage.output_tokens"] == 3
    assert gen.attributes["gen_ai.operation.name"] == "chat"
    assert gen.parent is not None


def test_pricing_override_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from llmops.pricing import load_pricing, reset_pricing_cache

    p = tmp_path / "pricing.json"
    p.write_text(
        json.dumps({"my-model": {"input_per_m": 1.0, "output_per_m": 2.0}}), encoding="utf-8"
    )
    monkeypatch.setenv("PRICING_PATH", str(p))
    reset_pricing_cache()
    assert estimate_cost("my-model", 1_000_000, 500_000) == pytest.approx(2.0)
    assert "openai/gpt-4o" not in load_pricing()
    monkeypatch.delenv("PRICING_PATH")
    reset_pricing_cache()
    assert estimate_cost("openai/gpt-4o", 1_000_000, 0) == pytest.approx(2.5)
    assert estimate_cost("rerank-v3.5", searches=10) == pytest.approx(0.02)


def test_jsonl_sink_per_process_file(tmp_path: Path):
    import os

    sink = JsonlSink(tmp_path / "spans.jsonl", per_process=True)
    with Tracer(sink).span("y"):
        pass
    sink.close()
    assert sink.path.name == f"spans-{os.getpid()}.jsonl"
    assert not (tmp_path / "spans.jsonl.lock").exists()
    assert [s.name for s in load_spans(tmp_path)] == ["y"]


def test_span_set_routes_unknown_keys_to_meta():
    s = Span(name="x").set(model="m", foo=1, meta={"bar": 2})
    assert s.model == "m" and s.meta == {"foo": 1, "bar": 2}


def test_wrap_client_uses_the_clients_provider_attribute_as_gen_ai_system():
    class Routed(FakeClient):
        provider = "openrouter"

    sink = MemorySink()
    t = Tracer(sink)
    wrap_client(Routed(), t).complete([Message(role="user", content="hi")])
    wrap_client(Routed(), t, system="custom").complete([Message(role="user", content="hi")])
    assert [s.meta["gen_ai.system"] for s in sink.spans] == ["openrouter", "custom"]


def test_fake_model_ids_used_by_sibling_repos_are_priced_at_zero():
    for model in (
        "fake-model",
        "fake-extractive-v1",
        "fake-hashed-bow-256",
        "fake-lexical-overlap",
    ):
        assert estimate_cost(model, 10_000, 10_000) == 0.0, model
