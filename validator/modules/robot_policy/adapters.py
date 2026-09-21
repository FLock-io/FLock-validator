"""Task adapters: the deterministic success predicate per task family.

WHY THIS FILE EXISTS
--------------------
`configs/tasks/*.yaml` already declares `task_family: standup`, and the README
already describes `TaskAdapter` as an abstraction -- but engine.run() hardcoded
`adapter=StandUp()` and nothing ever read `task_family`. Adding a second task
family meant editing the engine. This registry closes that gap: a new family is
a new class plus one line in TASK_ADAPTERS.

An adapter owns three things:
  * `required_fields` -- which trajectory keys it needs. Checked once against
    the first sample so a runner/adapter mismatch fails with a useful message
    instead of a KeyError 40 episodes in.
  * `evaluate(traj, cfg)` -> (success, completion_time_s, stable_fraction)
  * `metrics(episodes, timeout_s)` -> family-specific aggregate metrics.
    Success rate and its Wilson interval are generic and stay in the engine;
    "mean completion time" is not generic and belongs here.

POPULATION MATTERS MORE THAN NORMALISATION
------------------------------------------
A metric computed only over SUCCESSFUL episodes is not comparable between
policies with different success rates, and it rewards failing. Measured here:

    policy      success   mean_t(successes only)   normalised time score
    standup       0.46            1.075 s                  0.785
    walking       0.29            1.000 s                  0.800   <-- WORSE
                                                                       policy,
                                                                       BETTER
                                                                       score

Fail everything except one lucky fast episode and the time component goes UP.
So every metric declares the population it is computed over, and anything
conditioned on success is additionally published in an IMPUTED form: failures
are charged the full timeout, which makes it defined over all episodes,
monotone in capability, and impossible to game by failing.
"""
from __future__ import annotations

import statistics


class StandUp:
    """Get off the floor and hold a stable upright pose.

    Success: base height and tilt within limits CONTINUOUSLY for `stable_for_s`.
    `completion_time_s` is the moment that streak completes, so it includes the
    dwell -- it is "time to confirmed success", not "time to first upright".
    """

    family = "standup"
    required_fields = ("t", "base_height_m", "tilt_deg", "fallen")

    def evaluate(self, tr, cfg):
        p = cfg["success_predicate"]
        need = max(1, round(p["stable_for_s"] / p["sample_dt_s"]))
        streak = stable = 0
        done = None
        for s in tr:
            ok = (not s["fallen"]
                  and s["base_height_m"] >= p["min_base_height_m"]
                  and abs(s["tilt_deg"]) <= p["max_tilt_deg"])
            stable += ok
            streak = streak + 1 if ok else 0
            if streak >= need and done is None:
                done = s["t"]
        return done is not None, done, stable / len(tr)

    def metrics(self, eps, timeout_s=None):
        times = [e.completion_time_s for e in eps if e.success]
        out = {
            # population: successes only -- DIAGNOSTIC ONLY, never score on it
            "mean_success_completion_time_s": statistics.fmean(times) if times else None,
            "mean_stable_fraction": statistics.fmean([e.stable_fraction for e in eps]),
        }
        if timeout_s:
            # population: all episodes. A failure is charged the full timeout,
            # i.e. right-censored at the horizon. Lower is better, and failing
            # can no longer improve it.
            imputed = [e.completion_time_s if e.success else float(timeout_s)
                       for e in eps]
            out["mean_time_to_success_imputed_s"] = statistics.fmean(imputed)
        return out


def velocity_errors(tr, p):
    """Per-sample tracking error and commanded magnitude for a twist trajectory.

    The ONE place this is computed: the adapter's verdict, the runner's evidence
    and the browser smoke check (smoke.js, kept in step by hand) all use the
    same definition, so the numbers on the walkthrough board are comparable.

    `velocity_window_s` > 0 averages the body velocity over a trailing window
    before comparing it with the command. Measured 2026-09-05: a waddling biped's
    instantaneous body-frame velocity oscillates at stride frequency, so the raw
    per-sample error scored Pollen's own walking policy (0.11 m/s at a 0.25
    command, in two independent simulators) as "worse than standing still"
    (relative error 1.6-1.8). A 1 s window puts it at 0.5 -- walking at about
    half the commanded speed, which is what it does. Standing still is still
    exactly 1.0 either way. Window 0 (or absent) is the pre-v3 behaviour.
    """
    w = p.get("ang_weight", 0.5)
    win_s = float(p.get("velocity_window_s", 0.0) or 0.0)
    n = len(tr)
    if win_s > 0 and n > 1:
        dts = sorted(tr[i]["t"] - tr[i - 1]["t"] for i in range(1, n))
        dt = dts[n // 2 - 1] if dts[n // 2 - 1] > 0 else 0.02
        W = max(1, int(round(win_s / dt)))
    else:
        W = 1
    errs, mags = [], []
    sx = sy = sz = 0.0
    for i, s in enumerate(tr):
        sx += s["lin_vel_x"]; sy += s["lin_vel_y"]; sz += s["ang_vel_z"]
        if i >= W:
            o = tr[i - W]
            sx -= o["lin_vel_x"]; sy -= o["lin_vel_y"]; sz -= o["ang_vel_z"]
        k = min(i + 1, W)
        vx, vy, vz = sx / k, sy / k, sz / k
        errs.append(abs(vx - s["cmd_lin_vel_x"]) + abs(vy - s["cmd_lin_vel_y"])
                    + w * abs(vz - s["cmd_ang_vel_z"]))
        mags.append(abs(s["cmd_lin_vel_x"]) + abs(s["cmd_lin_vel_y"])
                    + w * abs(s["cmd_ang_vel_z"]))
    return errs, mags


class VelocityTracking:
    """Track a commanded body twist while staying upright.

    Success: mean absolute twist-tracking error stays under `max_mean_error`
    for the episode AND the robot never falls. `stable_fraction` here is the
    fraction of samples inside the per-step error tolerance, which keeps the
    field meaningful across families even though its definition differs.
    """

    family = "velocity"
    required_fields = ("t", "tilt_deg", "fallen",
                       "lin_vel_x", "lin_vel_y", "ang_vel_z",
                       "cmd_lin_vel_x", "cmd_lin_vel_y", "cmd_ang_vel_z")

    def evaluate(self, tr, cfg):
        p = cfg["success_predicate"]
        tol = p["per_step_error_tol"]
        errs, mags = velocity_errors(tr, p)
        good, fell = 0, False
        for s, e, m in zip(tr, errs, mags):
            # In-tolerance is scaled by what was asked for, for the same
            # reason the predicate is (below): a fixed absolute band hands
            # free credit to a motionless policy whenever the command is small.
            good += e <= max(tol, p.get("rel_step_tol", 0.0) * m)
            fell = fell or s["fallen"] or abs(s["tilt_deg"]) > p["max_tilt_deg"]
        mean_err = statistics.fmean(errs)
        mean_mag = statistics.fmean(mags)
        if fell:
            return False, None, good / len(tr)
        # RELATIVE error is the default bar. A policy that simply does not move
        # scores error == commanded magnitude, i.e. a relative error of 1.0,
        # WHATEVER the command was -- so the threshold means the same thing in
        # an episode commanding a brisk walk and one commanding a crawl, and
        # "did nothing" can no longer win the easy draws. `floor` keeps a
        # near-zero command from dividing a tiny error into a huge ratio.
        rel_cap = p.get("max_mean_relative_error")
        if rel_cap is not None:
            floor = p.get("cmd_magnitude_floor", 0.10)
            rel = mean_err / max(mean_mag, floor)
            return rel <= rel_cap, None, good / len(tr)
        # Absolute fallback, for suites frozen before the relative bar existed.
        return mean_err <= p["max_mean_error"], None, good / len(tr)

    def metrics(self, eps, timeout_s=None):
        # both populations are ALL episodes; no conditioning, nothing to impute
        out = {
            "mean_tracking_error": statistics.fmean(
                [e.evidence.get("mean_tracking_error", 0.0) for e in eps]),
            "mean_in_tolerance_fraction": statistics.fmean(
                [e.stable_fraction for e in eps]),
        }
        # Reported alongside the raw error because it is the one that is
        # comparable across episodes: 1.0 means "moved as little as standing
        # still would have", below 1.0 means the policy tracked the command.
        rel = [e.evidence["mean_tracking_error"]
               / max(e.evidence["mean_cmd_magnitude"], 0.10)
               for e in eps if e.evidence.get("mean_cmd_magnitude") is not None]
        if rel:
            out["mean_relative_tracking_error"] = statistics.fmean(rel)
        return out


TASK_ADAPTERS = {a.family: a for a in (StandUp(), VelocityTracking())}


def get_adapter(task_cfg):
    family = task_cfg.get("task_family")
    if family not in TASK_ADAPTERS:
        raise ValueError(
            f"unknown task_family {family!r}; known: {sorted(TASK_ADAPTERS)}. "
            "Add a class in adapters.py and register it in TASK_ADAPTERS.")
    return TASK_ADAPTERS[family]
