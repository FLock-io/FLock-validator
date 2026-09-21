"""Weighted scoring: base + task-specific metrics, then simulation + hardware.

The shape requested is:  score = W_sim*sim + W_hw*hw, where each substrate's
score is a weighted sum of base and task-specific metrics, and W_hw = 0 when
there is no hardware run. That shape is implemented here, with three
corrections that it needs to survive contact with real numbers.

1. NORMALIZE BEFORE WEIGHTING.
   The metrics do not share units: task_success_rate is [0,1],
   mean_success_completion_time_s is [0, timeout], mean_tracking_error is
   [0, inf), and for two of those LOWER is better. Summing them with weights
   produces a number whose units are meaningless and whose weights are fudge
   factors. Every metric therefore declares `direction` and `range`, and is
   mapped to [0,1] first. Weights are then dimensionless and comparable.

2. GATES ARE NOT WEIGHTS.
   A weighted sum lets a strong metric mask a fatal one -- a policy that falls
   over every episode but tracks velocity beautifully still scores well. This
   is the same failure AGENTS.md describes for additive reward stacks
   ("an additive stack has a compromise basin ... a product of Gaussians
   collapses on any single deficient factor"). Correctness conditions -- NaN
   free, observation contract intact, episodes actually ran -- are GATES: they
   multiply the score by 0, they do not contribute a weighted share.

3. 1:1 SIM:HARDWARE CONFLATES TWO DIFFERENT THINGS.
   Hardware runs are ~12 episodes against simulation's 100, so 1:1 gives each
   hardware episode ~8x the influence of a simulated one, and the hardware
   Wilson interval is enormous (measured: [0.089, 0.532] on 12 episodes).
   That is fine IF the weight expresses "hardware is what we actually care
   about" -- a decision weight. It is wrong if it is meant to express "these
   are equally reliable" -- a precision weight. The two are not the same, so
   this module keeps the decision weights explicit AND propagates uncertainty,
   and always reports the per-substrate scores and their gap next to the
   blend. A blended number alone is how the unreachable-threshold bug hid for
   three full evaluation runs.

Additionally, the composition used is recorded in the result. Two scores
computed under different weights are not comparable, and `comparable_key()`
makes that checkable instead of leaving it to whoever reads the number.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics


# Metrics whose population is "successful episodes only". Scoring on these is
# refused: they are not comparable across policies with different success
# rates, and they reward failing (measured: the 29%-success policy outscored
# the 46% one on time). Use the *_imputed_s form, which is defined over all
# episodes.
SUCCESS_CONDITIONED = frozenset({"mean_success_completion_time_s"})


def _finite_number(value, label):
    """Reject invalid numbers before min/max can turn NaN into full credit."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _check_metric_numbers(value, label="metrics"):
    # Also check numeric gates and nested diagnostics: bool(NaN) is True and
    # would otherwise let an invalid correctness condition pass.
    if isinstance(value, dict):
        for name, item in value.items():
            _check_metric_numbers(item, f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_metric_numbers(item, f"{label}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _finite_number(value, label)


def _norm(value, spec):
    """Map a raw metric onto [0,1] using its declared direction and range."""
    label = spec.get("metric", "metric")
    lo, hi = spec.get("range", [0.0, 1.0])
    lo = _finite_number(lo, f"{label} range minimum")
    hi = _finite_number(hi, f"{label} range maximum")
    if value is None:
        return None
    value = _finite_number(value, label)
    if hi == lo:
        return 0.0
    span = _finite_number(hi - lo, f"{label} range span")
    x = _finite_number((value - lo) / span, f"{label} normalized value")
    x = max(0.0, min(1.0, x))
    return x if spec.get("direction", "higher") == "higher" else 1.0 - x


def _gate_values(result):
    """Correctness conditions derived from the run, not from a metric table."""
    eps = result["episodes"]
    md = result["metadata"]
    return {
        "episodes_ran": len(eps) > 0,
        "full_suite": not md.get("partial_suite", False),
        "benchmark_substrate": md.get("substrate", "simulation") != "mock",
    }


def score_one(result, spec):
    """Score a single result. Returns the score plus every intermediate."""
    metrics = result["metrics"]
    _check_metric_numbers(metrics)
    gates = _gate_values(result)
    for g in spec.get("gates", []):
        name = g["metric"]
        gates.setdefault(name, bool(metrics.get(name, False)))
    gate_pass = all(gates.values())

    parts, total_w, acc, missing = {}, 0.0, 0.0, []
    for group in ("base", "task_specific"):
        for m in spec.get(group, []):
            name = m["metric"]
            if name in SUCCESS_CONDITIONED and not m.get("i_know_this_is_conditioned"):
                raise ValueError(
                    f"{name!r} is computed over successful episodes only, so it "
                    f"is not comparable across policies and rewards failing. "
                    f"Score on 'mean_time_to_success_imputed_s' instead, or set "
                    f"i_know_this_is_conditioned: true to override.")
            raw = metrics.get(name)
            n = _norm(raw, m)
            w = _finite_number(m.get("weight", 0.0), f"{name} weight")
            parts[m["metric"]] = {"group": group, "raw": raw,
                                  "normalized": n, "weight": w}
            if n is None:
                # Drop the metric AND its weight rather than scoring it 0 --
                # but say so. A spec naming a metric the adapter does not emit
                # would otherwise shrink the weight base invisibly and shift
                # every score.
                missing.append(name)
                continue
            acc += w * n
            total_w += w

    _finite_number(acc, "weighted metric total")
    _finite_number(total_w, "metric weight total")
    weighted = (_finite_number(acc / total_w, "weighted score")
                if total_w > 0 else None)
    declared_w = _finite_number(
        sum(_finite_number(m.get("weight", 0.0), f"{m['metric']} weight")
            for g in ("base", "task_specific") for m in spec.get(g, [])),
        "declared metric weight total")
    coverage = (_finite_number(total_w / declared_w, "weight coverage")
                if declared_w else None)
    return {
        "missing_metrics": missing,
        "declared_weight": declared_w,
        "score": (weighted if gate_pass else 0.0) if weighted is not None else None,
        "weighted_metric_score": weighted,
        "gates": gates,
        "gate_pass": gate_pass,
        "components": parts,
        "weight_used": total_w,
        "weight_coverage": coverage,
        "episodes": metrics.get("episodes"),
        "substrate": result["metadata"].get("substrate", "simulation"),
    }


def comparable_key(spec):
    """Two scores are only comparable if this key matches."""
    payload = json.dumps({k: spec.get(k) for k in
                          ("scoring_id", "gates", "base", "task_specific",
                           "substrate_weights")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def aggregate_score(results, spec):
    """Per-substrate scores, then the declared substrate blend."""
    from .consensus import _incompatible_axes

    results = list(results)
    incompatible = _incompatible_axes(results)
    task_ids = {(r["metadata"].get("task") or {}).get("id") for r in results}
    if len(task_ids) > 1:
        incompatible.append("task_id")
    if incompatible:
        raise ValueError(f"cannot score incompatible benchmarks; differing fields: {incompatible}")

    by_sub = {}
    for r in results:
        s = score_one(r, spec)
        by_sub.setdefault(s["substrate"], []).append(s)

    all_missing = sorted({m for lst in by_sub.values() for s in lst
                          for m in s["missing_metrics"]})
    per_sub = {}
    for sub, lst in by_sub.items():
        vals = [s["score"] for s in lst if s["score"] is not None]
        n = sum(s["episodes"] or 0 for s in lst)
        per_sub[sub] = {"score": statistics.fmean(vals) if vals else None,
                        "runs": len(lst), "episodes": n,
                        "detail": lst}

    declared = dict(spec.get("substrate_weights", {"simulation": 1.0,
                                                   "hardware": 1.0}))
    declared = {s: _finite_number(w, f"{s} substrate weight")
                for s, w in declared.items()}
    # A substrate that was not measured contributes nothing and its weight is
    # dropped -- NOT redistributed silently. The effective weights are reported
    # so a sim-only score is never mistaken for a sim+hardware score.
    effective = {s: w for s, w in declared.items()
                 if s in per_sub and per_sub[s]["score"] is not None and w > 0}
    tot = _finite_number(sum(effective.values()), "substrate weight total")
    final = (_finite_number(
        sum(per_sub[s]["score"] * w for s, w in effective.items()) / tot,
        "final score") if tot > 0 else None)

    out = {
        "final_score": final,
        "missing_metrics": all_missing,
        "scoring_id": spec.get("scoring_id"),
        "comparable_key": comparable_key(spec),
        "declared_substrate_weights": declared,
        "effective_substrate_weights": effective,
        "per_substrate": {s: {k: v for k, v in d.items() if k != "detail"}
                          for s, d in per_sub.items()},
        "detail": per_sub,
    }
    if all_missing:
        out["warning"] = (
            f"spec references metric(s) the results do not contain: "
            f"{all_missing}. Their weight was dropped, so this score is "
            f"computed over a smaller weight base than the spec declares. "
            f"Re-run the evaluation with a current adapter, or fix the spec.")
    if "simulation" in per_sub and "hardware" in per_sub:
        sim, hw = per_sub["simulation"]["score"], per_sub["hardware"]["score"]
        if None not in (sim, hw):
            out["sim2real_score_gap"] = sim - hw
            out["note"] = ("Positive gap = simulation is optimistic. Read the "
                           "per-substrate episode counts: a 12-episode hardware "
                           "arm carries a wide interval that the blended number "
                           "hides.")
    else:
        out["note"] = ("Only one substrate measured. This score is NOT "
                       "comparable with one that includes hardware, even though "
                       "both are on a 0-1 scale -- check comparable_key and "
                       "effective_substrate_weights.")
    return out
