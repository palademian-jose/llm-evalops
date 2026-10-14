from __future__ import annotations

import math

import pytest

from evalkit.metrics import (
    MetricContext,
    aggregate_scores,
    case_passed,
    compute_case_scores,
    get_metric,
    metric,
)
from evalkit.types import GoldenCase


def case(**kw) -> GoldenCase:
    return GoldenCase(
        id="c", input=kw.pop("input", {"question": "what is the capital of france"}), expected=kw
    )


def test_exact_match_and_f1():
    c = case(answer="The Eiffel Tower")
    ctx = MetricContext()
    assert get_metric("exact_match")(c, {"answer": "eiffel tower."}, ctx) is True
    assert get_metric("exact_match")(c, {"answer": "Louvre"}, ctx) is False
    assert get_metric("f1_token")(c, {"answer": "the tower"}, ctx) == pytest.approx(
        2 * 1 * 0.5 / 1.5
    )
    assert get_metric("f1_token")(case(), {"answer": "x"}, ctx) is None


def test_retrieval_metrics():
    c = case(relevant=["d1", "d3"])
    out = {"retrieved": ["d2", "d1", "d4", "d3", "d5"]}
    assert get_metric("recall_at_2")(c, out, MetricContext(k=10)) == 0.5
    assert get_metric("recall_at_k")(c, out, MetricContext(k=4)) == 1.0
    assert get_metric("precision_at_4")(c, out, MetricContext()) == 0.5
    assert get_metric("mrr")(c, out, MetricContext()) == 0.5
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3)
    assert get_metric("ndcg_at_5")(c, out, MetricContext()) == pytest.approx(dcg / idcg)
    assert get_metric("mrr")(c, {"retrieved": []}, MetricContext()) == 0.0
    assert get_metric("recall_at_k")(case(), out, MetricContext()) is None


def test_at_k_alias_requires_registered_base():
    with pytest.raises(KeyError):
        get_metric("nope_at_3")
    with pytest.raises(KeyError):
        get_metric("unknown")


def test_judge_backed_metrics_with_fake_provider():
    c = case(answer="paris")
    good = {
        "answer": "the capital of france is paris",
        "contexts": ["Paris is the capital of France."],
    }
    bad = {
        "answer": "berlin is a lovely city with museums",
        "contexts": ["Paris is the capital of France."],
    }
    ctx = MetricContext()
    assert get_metric("faithfulness")(c, good, ctx) is True
    assert get_metric("faithfulness")(c, bad, ctx) is False
    assert get_metric("faithfulness")(c, {"answer": "x"}, ctx) is None
    assert get_metric("answer_relevance")(c, good, ctx) is True
    assert (
        get_metric("context_precision")(
            c,
            {"answer": "x", "contexts": ["capital of france", "unrelated text about cheese"]},
            ctx,
        )
        == 0.5
    )


def test_task_success_and_pass_pow_k():
    c = case(answer="42")
    ctx = MetricContext(runs=[{"answer": "42"}, {"answer": "42"}, {"answer": "41"}])
    assert get_metric("task_success")(c, {"answer": "42"}, MetricContext()) is True
    assert (
        get_metric("task_success")(c, {"success": False, "answer": "42"}, MetricContext()) is False
    )
    assert get_metric("pass_pow_k")(c, {"answer": "42"}, ctx) is False
    ctx_all = MetricContext(runs=[{"answer": "42"}] * 3)
    assert get_metric("pass_pow_k")(c, {"answer": "42"}, ctx_all) is True
    custom = MetricContext(verifier=lambda case, out: out.get("steps", 99) <= 3)
    assert get_metric("task_success")(c, {"steps": 2}, custom) is True


def test_field_accuracy():
    c = case(fields={"name": "Acme Inc.", "amount": 12.5, "tags": ["a", "b"], "active": True})
    out = {"fields": {"name": "acme inc", "amount": "12.50", "tags": ["b", "a"], "active": False}}
    assert get_metric("field_accuracy")(c, out, MetricContext()) == 0.75
    assert get_metric("field_accuracy")(case(), out, MetricContext()) is None


def test_registry_plugin_and_aggregation():
    @metric("always_half", description="constant")
    def always_half(case, output, ctx):
        return 0.5

    c = case(answer="x")
    scores = compute_case_scores(
        c, {"answer": "x"}, ["exact_match", "always_half"], MetricContext()
    )
    assert scores == {"exact_match": True, "always_half": 0.5}
    agg = aggregate_scores(
        ["exact_match", "always_half", "recall_at_k"],
        [scores, {"exact_match": False, "always_half": 0.5}],
    )
    assert agg == {"exact_match": 0.5, "always_half": 0.5}
    assert case_passed({"a": True, "b": 0.7}) and not case_passed({"a": True, "b": 0.2})
    assert not case_passed({})
