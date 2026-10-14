"""Built-in metrics. Precise definitions live in each docstring and in README "Metric definitions".

Field conventions (all optional unless the metric says otherwise):
- `output["answer"]`, `expected["answer"]`: free-text answers.
- `output["retrieved"]`: ranked list of document ids; `expected["relevant"]`: set of relevant ids.
- `output["contexts"]`: list of context strings handed to the generator.
- `output["fields"]`, `expected["fields"]`: extraction dicts.
- `output["success"]`: optional boolean a target may set; `expected["answer"]` otherwise.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from collections.abc import Callable
from typing import Any

from evalkit.metrics.registry import MetricContext, metric
from evalkit.types import GoldenCase, Score

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: Any) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation and articles, collapse spaces."""
    s = str(text if text is not None else "").lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _answer(d: dict[str, Any]) -> str:
    v = d.get("answer")
    return "" if v is None else str(v)


def _question(case: GoldenCase) -> str:
    for key in ("question", "query", "task", "prompt", "input", "text"):
        if key in case.input:
            return str(case.input[key])
    return " ".join(str(v) for v in case.input.values())


# ---- lexical -----------------------------------------------------------------------------------


@metric("exact_match", kind="bool")
def exact_match(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """1 when normalize(output.answer) == normalize(expected.answer); None without expected."""
    if "answer" not in case.expected:
        return None
    return normalize_answer(_answer(output)) == normalize_answer(case.expected["answer"])


@metric("f1_token")
def f1_token(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """SQuAD token F1: harmonic mean of token precision and recall over normalized answers.

    Multiset overlap (`Counter & Counter`); 0 when either side is empty unless both are.
    """
    if "answer" not in case.expected:
        return None
    pred = normalize_answer(_answer(output)).split()
    gold = normalize_answer(case.expected["answer"]).split()
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


# ---- retrieval ---------------------------------------------------------------------------------


def _ranked(output: dict[str, Any]) -> list[str]:
    return [str(x) for x in output.get("retrieved") or []]


def _relevant(case: GoldenCase) -> set[str]:
    return {str(x) for x in case.expected.get("relevant") or []}


@metric("recall_at_k")
def recall_at_k(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """|relevant ∩ retrieved[:k]| / |relevant|. None when the case has no relevant ids."""
    rel = _relevant(case)
    if not rel:
        return None
    top = set(_ranked(output)[: ctx.k])
    return len(rel & top) / len(rel)


@metric("precision_at_k")
def precision_at_k(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """|relevant ∩ retrieved[:k]| / k (denominator is k, not the number returned)."""
    rel = _relevant(case)
    if not rel:
        return None
    top = _ranked(output)[: ctx.k]
    return len([d for d in top if d in rel]) / ctx.k


@metric("mrr")
def mrr(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """1 / rank of the first relevant id in the full ranked list; 0 when none is retrieved."""
    rel = _relevant(case)
    if not rel:
        return None
    for i, d in enumerate(_ranked(output), start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


@metric("ndcg_at_k")
def ndcg_at_k(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Binary-relevance nDCG@k: DCG = Σ rel_i / log2(i + 1), normalized by the ideal DCG."""
    rel = _relevant(case)
    if not rel:
        return None
    top = _ranked(output)[: ctx.k]
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(top, start=1) if d in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), ctx.k) + 1))
    return dcg / ideal if ideal else 0.0


# ---- judge-backed ------------------------------------------------------------------------------


def _contexts(output: dict[str, Any]) -> list[str]:
    return [str(c) for c in output.get("contexts") or []]


@metric("faithfulness", kind="bool", needs_judge=True)
def faithfulness(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Judge verdict: every claim in output.answer is supported by output.contexts.

    None when the output has no contexts (nothing to be faithful to).
    """
    ctxs = _contexts(output)
    if not ctxs:
        return None
    verdict = ctx.judge("faithfulness")(
        {"question": _question(case)}, {"answer": _answer(output)}, {"contexts": ctxs}
    )
    return verdict.pass_


@metric("answer_relevance", kind="bool", needs_judge=True)
def answer_relevance(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Judge verdict: output.answer directly addresses the question (ignoring correctness)."""
    verdict = ctx.judge("answer_relevance")(
        {"question": _question(case)}, {"answer": _answer(output)}, None
    )
    return verdict.pass_


@metric("context_precision", needs_judge=True)
def context_precision(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Fraction of output.contexts the judge deems relevant to the question (one call per context).

    None when there are no contexts.
    """
    ctxs = _contexts(output)
    if not ctxs:
        return None
    judge = ctx.judge("context_relevance")
    hits = sum(1 for c in ctxs if judge({"question": _question(case)}, {"context": c}, None).pass_)
    return hits / len(ctxs)


# ---- task / agentic ----------------------------------------------------------------------------


def default_verifier(case: GoldenCase, output: dict[str, Any]) -> bool:
    """`output.success` when present, else normalized exact match on `expected.answer`."""
    if isinstance(output.get("success"), bool):
        return output["success"]
    if "answer" in case.expected:
        return normalize_answer(_answer(output)) == normalize_answer(case.expected["answer"])
    return False


def _verifier(ctx: MetricContext) -> Callable[[GoldenCase, dict[str, Any]], bool]:
    return ctx.verifier or default_verifier


@metric("task_success", kind="bool")
def task_success(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Boolean from the verifier callable (`--verifier module:callable`; default_verifier)."""
    return bool(_verifier(ctx)(case, output))


@metric("pass_pow_k", kind="bool")
def pass_pow_k(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """pass^k: ALL k independent runs of the case succeed (verifier). Not pass@k.

    pass@k ("at least one of k runs succeeds") rewards lucky retries and rises with k;
    pass^k ("every one of k runs succeeds") measures reliability and falls with k. For a
    system that must be right every time, pass^k is the number to gate on. With a single run
    (no `--k`), pass^1 == task_success.
    """
    runs = ctx.runs if ctx.runs else [output]
    verify = _verifier(ctx)
    return all(bool(verify(case, r)) for r in runs)


# ---- extraction --------------------------------------------------------------------------------


@metric("field_accuracy")
def field_accuracy(case: GoldenCase, output: dict[str, Any], ctx: MetricContext) -> Score:
    """Fraction of expected.fields keys whose normalized value equals output.fields[key].

    Lists compare as sets of normalized strings; numbers compare with 1e-9 tolerance.
    None when expected.fields is missing or empty.
    """
    exp = case.expected.get("fields")
    if not isinstance(exp, dict) or not exp:
        return None
    got = output.get("fields") if isinstance(output.get("fields"), dict) else {}
    hits = 0
    for key, want in exp.items():
        have = got.get(key)
        if _field_equal(want, have):
            hits += 1
    return hits / len(exp)


def _field_equal(want: Any, have: Any) -> bool:
    if isinstance(want, int | float) and not isinstance(want, bool):
        try:
            return abs(float(want) - float(have)) < 1e-9
        except (TypeError, ValueError):
            return False
    if isinstance(want, list):
        if not isinstance(have, list):
            return False
        return {normalize_answer(x) for x in want} == {normalize_answer(x) for x in have}
    if isinstance(want, bool):
        return want is have or (isinstance(have, str) and have.lower() == str(want).lower())
    return normalize_answer(want) == normalize_answer(have)
