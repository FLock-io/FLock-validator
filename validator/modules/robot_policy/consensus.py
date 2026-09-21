"""Cross-validation: what aggregating independent validators can and cannot say.

--------------------------------------------------------------------------
THE PROBLEM THIS REPLACES
--------------------------------------------------------------------------
The original `aggregate()` reported `reproducibility_agreement = 1 - disagreements/n`
and required same_manifest / same_policy / same_suite. Run against two real
validators it reported perfect agreement -- correctly, and uselessly:

  * NOTHING had varied. Same code, same suite, same seeds, same GPU. Agreement
    there measures DETERMINISM of the stack, not independent validation of the
    policy. Calling it "consensus" overstates it by a lot.
  * Both validators scored 0/100, and two validators that always answer "fail"
    agree 100% of the time. Raw agreement is satisfied by chance whenever the
    outcome is degenerate.

So this module does three things the old one did not:

  1. Works out WHAT VARIED between validators, and classifies the comparison
     accordingly. A replication is reported as a replication, not as consensus.
  2. Corrects agreement for chance (Fleiss' kappa). Degenerate all-pass or
     all-fail sets produce an undefined kappa, which is the honest answer:
     the comparison carries no information about the policy.
  3. Separates substrates. Simulator results and hardware results are never
     pooled into one success rate; they are compared, and their difference is
     reported as the sim2real transfer gap.
"""
from __future__ import annotations

import json
import math
import statistics

from .substrate import HARDWARE, SIMULATION, is_benchmark_valid

# Fields whose variation makes a comparison more (or less) independent.
# Order matters: the first axis that varies determines the classification.
INDEPENDENCE_AXES = (
    "policy_sha256",     # different policy   -> not a comparison at all
    "substrate",         # sim vs real        -> transfer, not consensus
    "task_family",       # different task     -> not comparable
    "task_config_hash",  # different predicate or command envelope
    "embodiment",        # different robot configuration
    "suite_family",      # different task distribution
    "suite_hash",        # different scenario draw (same family = independent draw)
    "simulator_commit",  # different upstream build
    # Same task code can still be a different physics build. Two results from
    # the same commit but different mujoco-warp are not replications of each
    # other, and nothing else here would notice.
    "physics_stack",     # different mjlab/mujoco/warp/torch versions
    "device",            # different GPU / driver
    "build_seed",        # different construction randomization
    "batch",             # different batch layout changes reset RNG draws
    "onnx_provider",     # different inference backend
)

# These fields identify the quantity being measured. Different simulator
# stacks or independent draws may be compared; different policies, predicates,
# robots or scenario distributions must not share an aggregate success rate.
BENCHMARK_AXES = ("policy_sha256", "task_family", "task_config_hash",
                  "embodiment", "suite_family")

# The packages that decide what the physics does. `microduck_rl_path` is
# deliberately excluded -- it is where someone happened to check the repo out,
# and including it would make every cross-machine comparison look like it
# varied a physics axis when nothing physical changed. `microduck_rl_commit`
# is excluded too: it is already the `simulator_commit` axis, and an axis that
# duplicates another one double-counts the same difference.
_STACK_KEYS = ("mjlab", "mjlab_microduck", "mujoco", "mujoco_warp", "warp",
               "torch", "onnxruntime")


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def _axis(meta, name):
    if name == "suite_family":
        return meta.get("suite_family") or meta.get("suite_id")
    if name == "simulator_commit":
        c = ((meta.get("simulator_stack") or {}).get("microduck_rl_commit")
             or (meta.get("simulator") or {}).get("source_commit"))
        # Results written before provenance was recorded carry the literal
        # placeholder "local", which is not an identity: treating it as one
        # would report two genuinely different simulators as the same build.
        return None if c in (None, "", "local", "unknown") else c
    if name == "physics_stack":
        st = meta.get("simulator_stack")
        if not st:
            return None
        return " ".join(f"{k}={st[k]}" for k in _STACK_KEYS if k in st)
    if name == "task_family":
        return meta.get("task_family")
    if name == "embodiment":
        value = meta.get(name)
        return json.dumps(value, sort_keys=True) if value is not None else None
    return meta.get(name)


def varied_axes(results):
    """Which independence axes differ across the given results."""
    out = {}
    for ax in INDEPENDENCE_AXES:
        vals = {_axis(r["metadata"], ax) for r in results}
        out[ax] = sorted(v for v in vals if v is not None) if len(vals) > 1 else None
    return {k: v for k, v in out.items() if v}


def classify(results):
    """What kind of comparison is this, and what may it be used to claim?"""
    v = varied_axes(results)
    if len(results) < 2:
        return "single", "One validator. Nothing is being cross-checked."
    if "policy_sha256" in v:
        return "different-policies", (
            "Validators evaluated DIFFERENT policies. This is a comparison of "
            "policies, not a consensus check; agreement is meaningless here.")
    if "task_family" in v:
        return "incomparable", "Different task families; results are not comparable."
    if any(a in v for a in ("task_config_hash", "embodiment")):
        return "incomparable", "Different task predicates or robots; results are not comparable."
    if "suite_family" in v:
        return "cross-distribution", (
            "Different task distributions. Aggregate per family, not across.")
    if "substrate" in v:
        return "cross-substrate", (
            "Simulator and hardware results are present. Do NOT pool them: "
            "their difference IS the sim2real transfer gap, reported separately.")
    if "suite_hash" in v:
        return "independent-draw", (
            "Same distribution, different scenario draw. Agreement here is "
            "evidence about GENERALIZATION, and the pooled rate is a stronger "
            "estimate than any single suite.")
    if any(a in v for a in ("simulator_commit", "physics_stack", "device",
                           "build_seed", "batch", "onnx_provider")):
        return "cross-stack", (
            "Same suite on a different build/device/seed. Agreement is evidence "
            "that the result is not an artifact of one stack.")
    return "replication", (
        "Nothing varied: same policy, same suite, same seeds, same stack. This "
        "is a DETERMINISM check of the harness, not independent validation. "
        "Agreement here says nothing about the policy.")


def fleiss_kappa(per_scenario):
    """Chance-corrected agreement over binary outcomes.

    per_scenario: list of lists of bools (one inner list per scenario).
    Returns None when every rater gave the same category everywhere -- with a
    degenerate outcome, chance agreement is already 1.0 and kappa is 0/0. That
    is the correct answer: raw agreement of 1.0 carries no information.
    """
    rows = [r for r in per_scenario if len(r) > 1]
    if not rows:
        return None
    n = len(rows[0])
    if any(len(r) != n for r in rows):
        return None
    p_bar = statistics.fmean(
        (sum(r) ** 2 + (n - sum(r)) ** 2 - n) / (n * (n - 1)) for r in rows)
    p_success = statistics.fmean(sum(r) / n for r in rows)
    p_e = p_success ** 2 + (1 - p_success) ** 2
    if abs(1 - p_e) < 1e-12:
        return None  # degenerate: everyone always said the same thing
    return (p_bar - p_e) / (1 - p_e)


def dispersion(results):
    """Do validators disagree more than sampling noise alone explains?

    Within-validator variance of a success rate is binomial, p(1-p)/n. If the
    variance BETWEEN validators is much larger, they are not measuring the same
    quantity -- different stacks, different suites, or a broken harness.
    """
    rates = [r["metrics"]["task_success_rate"] for r in results]
    ns = [r["metrics"]["episodes"] for r in results]
    if len(rates) < 2:
        return None
    within = statistics.fmean(p * (1 - p) / n for p, n in zip(rates, ns) if n)
    between = statistics.variance(rates)
    return {
        "between_validator_var": between,
        "expected_sampling_var": within,
        "ratio": (between / within) if within > 0 else None,
        "note": ("between-validator spread exceeds sampling noise; validators are "
                 "not measuring the same quantity"
                 if within > 0 and between > 3 * within else
                 "spread is consistent with sampling noise"),
    }


def _pool(results):
    k = sum(r["metrics"]["successes"] for r in results)
    n = sum(r["metrics"]["episodes"] for r in results)
    return {"episodes": n, "successes": k,
            "task_success_rate": (k / n) if n else None,
            "task_success_ci95": wilson(k, n)}


def _incompatible_axes(results):
    return [name for name in BENCHMARK_AXES
            if len({_axis(r["metadata"], name) for r in results}) > 1]


def _substrate_summary(results):
    incompatible = _incompatible_axes(results)
    if not incompatible:
        return _pool(results)
    # Preserve useful individual results without producing a number that mixes
    # unrelated benchmarks. Existing homogeneous summaries keep their schema.
    return {"pooled": False, "incompatible_axes": incompatible,
            "runs": [{"metadata": r["metadata"], **_pool([r])} for r in results]}


def consensus(results):
    """Aggregate validator results into a defensible statement."""
    if not results:
        return {"validator_count": 0, "error": "no results given"}

    for r in results:
        r["metadata"].setdefault("substrate", SIMULATION)

    invalid = [r["metadata"]["validator_id"] for r in results
               if not is_benchmark_valid(r["metadata"]["substrate"])]

    kind, meaning = classify(results)
    incompatible = _incompatible_axes(results)
    maps = [{e["scenario_id"]: bool(e["success"]) for e in r["episodes"]}
            for r in results]
    # A scenario's displayed ID is not its identity: changing the suite seed
    # can leave IDs unchanged. Agreement needs the same frozen scenario draw.
    same_draw = len({_axis(r["metadata"], "suite_hash") for r in results}) == 1
    common = (sorted(set.intersection(*(set(m) for m in maps)))
              if maps and same_draw and not incompatible else [])
    per_scen = [[m[s] for m in maps] for s in common]
    raw = (statistics.fmean(len(set(v)) == 1 for v in per_scen)
           if per_scen else None)
    kappa = fleiss_kappa(per_scen)

    by_sub = {}
    for r in results:
        by_sub.setdefault(r["metadata"]["substrate"], []).append(r)

    out = {
        "validator_count": len(results),
        "comparison_kind": kind,
        "what_this_can_claim": meaning,
        "varied_axes": varied_axes(results),
        "substrates": {s: len(v) for s, v in by_sub.items()},
        "benchmark_valid": not invalid and not incompatible,
        "incompatible_axes": incompatible,
        "raw_agreement": raw,
        "chance_corrected_agreement_kappa": kappa,
        "agreement_is_informative": kappa is not None,
        "common_scenarios": len(common),
        "dispersion": dispersion(results) if not incompatible else None,
        "per_substrate": {s: _substrate_summary(v) for s, v in by_sub.items()},
    }
    errors = []
    if invalid:
        errors.append(f"non-benchmark substrate in {invalid}; "
                      "mock results must not back a claim")
    if incompatible:
        errors.append(f"incompatible benchmarks differ on {incompatible}; "
                      "no combined benchmark claim or sim2real gap is valid")
    if errors:
        out["error"] = "; ".join(errors)
    if kappa is None and raw is not None:
        out["agreement_note"] = (
            "Raw agreement is degenerate: every validator returned the same "
            "outcome on every scenario, so chance agreement is already 1.0 and "
            "kappa is undefined. This comparison carries no information about "
            "the policy.")

    # Only sim vs real gets a transfer gap; pooling them would be wrong.
    if SIMULATION in by_sub and HARDWARE in by_sub and not incompatible:
        sim, hw = _pool(by_sub[SIMULATION]), _pool(by_sub[HARDWARE])
        out["sim2real"] = {
            "simulation_rate": sim["task_success_rate"],
            "hardware_rate": hw["task_success_rate"],
            "transfer_gap": (None if None in (sim["task_success_rate"], hw["task_success_rate"])
                             else sim["task_success_rate"] - hw["task_success_rate"]),
            "hardware_ci95": hw["task_success_ci95"],
            "note": ("Positive gap = simulator is optimistic. Hardware episode "
                     "counts are small by nature; read the CI, not the point "
                     "estimate."),
        }
    return out
