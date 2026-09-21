"""RobotPolicy model download, evaluation and metrics for the existing validator."""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import yaml

from .engine import Manifest, hobj, run
from .errors import InvalidSubmission, ModelParameterLimit, LOWEST_POSSIBLE_REWARD
from .robot_policy_runner import RobotPolicyRunner
from .policy import check_policy, external_files, load_policy_metadata, repo_filename, stage_policy
from .suite import SuiteError, fetch_suite, validate_suite_url


class NotConfigured(RuntimeError):
    """A setting the arena needs is missing (API key, task id, ...)."""


_SUBMITTER_EXC = ("RevisionNotFoundError", "EntryNotFoundError", "RemoteEntryNotFoundError",
                  "HFValidationError", "InvalidProtobuf", "InvalidGraph")


_LOCAL_EXC = ("RepositoryNotFoundError", "GatedRepoError", "LocalEntryNotFoundError")


_OURS_HINT = ("out of memory", "cuda", "no cuda", "modulenotfounderror", "importerror",
              "no space left", "connection", "timed out", "timeout", "temporary failure",
              "permission denied", "killed", "mjlab", "warp")


_SUBMITTER_HINT = ("off-contract", "not a valid onnx", "invalid protobuf",
                   "exceed limit", "no such file in repo",
                   "does not appear to have a file named")


def classify_failure(exc: BaseException) -> str:
    """'submitter' | 'ours' | 'unknown'. Only 'submitter' may fail an assignment."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if isinstance(exc, InvalidSubmission):
        return "submitter"
    if isinstance(exc, (ImportError, NotConfigured, SuiteError)) or name in _LOCAL_EXC:
        return "ours"
    if name in _SUBMITTER_EXC:
        return "submitter"
    if isinstance(exc, OSError):
        return "ours"
    if any(h in msg for h in _OURS_HINT):
        return "ours"
    if any(h in msg for h in _SUBMITTER_HINT):
        return "submitter"
    return "unknown"


def assignment_metrics(result: dict, spec: dict | None = None) -> dict:
    """What a validator reports for an assignment.

    FLock's RL module reports one number, `average_reward`. Ours is the pass
    rate (or the composed score when a scoring spec is given) so the ledger's
    ranking works unchanged; the interval, the tracking error, the suite and
    the policy hash ride along so the number can be audited.
    """
    m = result.get("metrics") or {}
    meta = result.get("metadata") or {}
    if meta.get("substrate") not in ("simulation", "hardware"):
        raise ValueError("only a simulation or hardware result may be reported")
    if meta.get("partial_suite"):
        raise ValueError("a partial suite may not be reported as a completed assignment")
    episodes = result.get("episodes") or []
    if (not isinstance(episodes, list) or not episodes
            or type(m.get("episodes")) is not int or m["episodes"] != len(episodes)):
        raise ValueError("episode evidence is missing or does not match the metric count")
    if meta.get("episodes_requested", len(episodes)) != len(episodes):
        raise ValueError("the result does not cover the full requested suite")
    rate = m.get("task_success_rate")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("task_success_rate must be a finite number in [0, 1]")
    if any(not isinstance(episode, dict) or type(episode.get("success")) is not bool
           for episode in episodes):
        raise ValueError("every episode requires a boolean success result")
    successes = sum(episode["success"] for episode in episodes)
    if (type(m.get("successes")) is not int or m["successes"] != successes
            or not math.isclose(rate, successes / len(episodes), rel_tol=0, abs_tol=1e-12)):
        raise ValueError("success metrics do not match the episode evidence")
    out = {"average_reward": float(rate),
           "task_success_rate": m.get("task_success_rate"),
           "task_success_ci95": m.get("task_success_ci95"),
           "episodes": m.get("episodes"),
           "policy_sha256": meta.get("policy_sha256"), "suite_id": meta.get("suite_id"),
           "suite_hash": meta.get("suite_hash"), "task_config_hash": meta.get("task_config_hash"),
           "build_seed": meta.get("build_seed"), "substrate": meta.get("substrate"),
           "task_config": Path((meta.get("task") or {}).get("config", "")).name if isinstance(meta.get("task"), dict) else None,
           "batch": meta.get("batch"),
           # The stack is useful provenance; the local filesystem path it also
           # carries is nobody else's business and goes to a shared ledger.
           "physics_stack": {k: v for k, v in (meta.get("simulator_stack") or {}).items()
                             if not k.endswith("_path")} or None}
    for k in ("mean_relative_tracking_error", "mean_tracking_error", "mean_in_tolerance_fraction",
              "mean_stable_fraction", "mean_time_to_success_imputed_s"):
        if m.get(k) is not None:
            out[k] = m[k]
    if spec is not None:
        from .scoring import aggregate_score
        sc = aggregate_score([result], spec)
        # aggregate_score returns `final_score`. Reading "score" produced None,
        # which fell through to 0.0 -- the arena ranks on this number, so every
        # spec-scored policy would have been reported as a total failure.
        composed = sc.get("final_score")
        if composed is None or not math.isfinite(composed) or sc.get("missing_metrics"):
            raise ValueError("scoring spec cannot produce a complete finite assignment score")
        out.update({"average_reward": float(composed), "score": composed})
        out.update({"scoring_id": sc.get("scoring_id"), "comparable_key": sc.get("comparable_key")})
    out = {k: v for k, v in out.items() if v is not None}
    json.dumps(out, allow_nan=False)
    return out


def fetch_policy(hg_repo_id: str, model_filename: str, revision: str, token: str | None = None) -> Path:
    """Download the submitted ONNX exactly as FLock's RL module does."""
    from huggingface_hub import hf_hub_download  # optional dependency
    from .policy import external_files, load_policy_metadata, repo_filename
    model_filename = repo_filename(model_filename)
    token = token or os.environ.get("HF_TOKEN") or None
    path = Path(hf_hub_download(hg_repo_id, model_filename, revision=revision, token=token))
    for filename in external_files(load_policy_metadata(path), model_filename):
        hf_hub_download(hg_repo_id, filename, revision=revision, token=token)
    return path


def write_manifest(path: Path, policy: Path, task_id_mjlab: str, task_config: Path,
                   suite: Path, suite_id: str, revision: str,
                   external_files: list[str] | None = None) -> Path:
    """A manifest for an arena assignment: same file our own runs use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep HF snapshot paths intact: resolving their symlinks would move the
    # model to blobs/, away from the external weights located beside it.
    def rel(p):
        absolute = Path(p).absolute()
        try:
            return os.path.relpath(absolute, path.parent)
        except ValueError:
            # Windows model caches and benchmarks can be on different drives.
            return str(absolute)
    import yaml
    path.write_text(yaml.safe_dump({
        "model": {"format": "onnx", "path": rel(policy),
                  **({"external_files": external_files} if external_files else {})},
        "embodiment": {"type": "microduck", "version": "v1"},
        "task": {"id": task_id_mjlab, "config": rel(task_config)},
        "simulator": {"backend": "mjlab", "upstream": "pollen-robotics/microduck_rl",
                      "source_commit": "unknown", "physics": "mujoco-warp", "hf_revision": revision},
        "suite": {"id": suite_id, "path": rel(suite)},
    }, sort_keys=False))
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def validate_input(data):
    for name in ("hg_repo_id", "model_filename", "revision"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise InvalidSubmission(f"assignment requires {name}")
    repo_filename(data["model_filename"])
    if not re.fullmatch(r"[0-9a-fA-F]{40}", data["revision"]):
        raise NotConfigured("assignment revision must be an immutable HF commit")
    limit = data.get("max_params")
    if type(limit) is not int or limit <= 0:
        raise NotConfigured("assignment max_params must be a positive integer")
    validate_suite_url(data.get("validation_set_url"))


def benchmark_config(config):
    """Read local task and scoring rules; FedLedger owns the suite selection."""
    task_path = Path(config["task_config"]).resolve()
    task = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise NotConfigured("task config must be an object")
    from .adapters import get_adapter
    get_adapter(task)
    expected_tasks = {"velocity": "Mjlab-Velocity-Flat-MicroDuck",
                      "standup": "Mjlab-StandUp-Flat-MicroDuck"}
    if task.get("mjlab_task_id") != expected_tasks[task["task_family"]]:
        raise NotConfigured("task family and mjlab_task_id do not match")
    if type(config.get("batch", 1)) is not int or config.get("batch", 1) < 1:
        raise NotConfigured("batch must be positive")
    spec = None
    if config.get("scoring_spec"):
        spec = yaml.safe_load(Path(config["scoring_spec"]).read_text(encoding="utf-8"))
        if not isinstance(spec, dict) or not spec.get("scoring_id"):
            raise NotConfigured("scoring spec needs a scoring_id")
    return task_path, task, spec


def evaluate_assignment(data, config, out, validator):
    task_path, task, spec = benchmark_config(config)
    validate_input(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    policy = fetch_policy(data["hg_repo_id"], data["model_filename"], data["revision"])
    policy = stage_policy(policy, out / "policy")
    try:
        params = check_policy(policy, data.get("max_params"))
    except ModelParameterLimit:
        # The reference RL evaluator completes oversized models with this penalty.
        return {"average_reward": LOWEST_POSSIBLE_REWARD}, None
    suite_path = out / "suite.yaml"
    suite = fetch_suite(data["validation_set_url"], suite_path,
                        task_id=task["mjlab_task_id"])
    manifest = write_manifest(out / "manifest.yaml", policy, task["mjlab_task_id"],
                              task_path, suite_path, suite["suite_id"], data["revision"],
                              external_files=external_files(load_policy_metadata(policy), policy.name))
    expected_policy_hash = Manifest(manifest).policy_hash
    runner = RobotPolicyRunner(task_id=task["mjlab_task_id"], device=config.get("device", "cuda:0"),
                        batch=config.get("batch", 1), build_seed=config.get("build_seed", 0),
                        onnx_provider=config.get("onnx_provider", "cpu"))
    try:
        result = run(manifest, validator, runner, progress_path=out / "progress.json")
    finally:
        runner.close()
    metrics = assignment_metrics(result, spec)
    if result["metrics"]["episodes"] != suite["episodes"]:
        raise NotConfigured("evaluation did not complete the assigned suite")
    if [episode.get("scenario_id") for episode in result["episodes"]] != [
            f"{suite['suite_id']}-{index + 1:06d}" for index in range(suite["episodes"])]:
        raise NotConfigured("evaluation scenarios do not match the assigned suite")
    if (result["metadata"].get("suite_id") != suite["suite_id"]
            or result["metadata"].get("suite_hash") != hobj(suite)
            or result["metadata"].get("task_config_hash") != hobj(task)
            or result["metadata"].get("policy_sha256") != expected_policy_hash):
        raise NotConfigured("evaluation result does not match its assignment and benchmark")
    if "validation_phase" in suite:
        result["metadata"]["validation_phase"] = suite["validation_phase"]
    write_json(out / "result.json", result)
    metrics["model_params"] = params
    return metrics, result
