import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub import errors as hf_errors

from validator.modules.robotics_vla.adapter import (
    count_policy_parameters,
    load_policy_from_adapter,
    resolve_model_dir,
)
from validator.modules.robotics_vla import (
    RoboticsVLAConfig,
    RoboticsVLAInputData,
    RoboticsVLAValidationModule,
)
from validator.modules.robotics_vla.errors import RoboticsSubmissionError
from validator.modules.robotics_vla.data_package import (
    resolve_validation_data_package,
    write_validation_package,
)
from validator.modules.robotics_vla.domain_randomization import (
    DOMAIN_RANDOMIZATION_VERSION,
    apply_domain_randomization_to_manifest,
)
from validator.modules.robotics_vla.manifest import EpisodeSpec, ValidationManifest
from validator.modules.robotics_vla.manifest import load_manifest
from validator.modules.robotics_vla.simulation import (
    RolloutSettings,
    adapt_action_for_env,
    compute_episode_score,
    compute_progress_score,
    compute_weighted_episode_score,
    compute_weighted_loss,
    difficulty_weight,
    extract_frame,
    prepare_policy_obs,
    query_policy_action,
    resolve_task_definition,
    target_nut_on_matching_peg_success,
    target_object_in_matching_bin_success,
    validate_action,
)
from validator.modules.robotics_vla.task_registry import (
    TASK_REGISTRY_VERSION,
    validate_task_registry_spec,
)


def test_manifest_loads_json_file(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "task": "lift_cube",
                        "instruction": "lift the cube",
                        "seed": 1,
                    }
                ],
            }
        )
    )

    manifest = load_manifest(str(manifest_path))

    assert manifest.suite_version == "panda_tabletop_v1"
    assert manifest.episodes[0].task == "lift_cube"


def test_validate_action_clips_and_rejects_bad_shapes():
    action = validate_action(np.array([2, -2, 0, 0, 0, 0, 0]), 7)

    assert action.tolist() == [1, -1, 0, 0, 0, 0, 0]

    with pytest.raises(RoboticsSubmissionError, match="shape") as excinfo:
        validate_action(np.zeros((6,), dtype=np.float32), 7)
    assert excinfo.value.failure_mode == "invalid_action"


def test_validate_action_rejects_nan():
    with pytest.raises(RoboticsSubmissionError, match="NaN") as excinfo:
        validate_action(np.array([0, 0, np.nan, 0, 0, 0, 0]), 7)
    assert excinfo.value.failure_mode == "invalid_action"


def test_adapt_action_for_env_matches_declared_env_dimension():
    class Env:
        action_dim = 6

    action = np.arange(7, dtype=np.float32)

    adapted = adapt_action_for_env(action, Env())

    np.testing.assert_array_equal(adapted, np.arange(6, dtype=np.float32))


def test_hybrid_strict_scoring_caps_failed_partial_credit():
    assert compute_episode_score(success=True, progress_score=0.0) == 1.0
    assert compute_progress_score(success=True, best_reward=0.0) == 1.0
    assert compute_episode_score(success=False, progress_score=1.0) == 0.40
    assert compute_episode_score(success=False, progress_score=0.5) == 0.20
    assert compute_progress_score(success=False, best_reward=2.0) == 1.0
    assert compute_progress_score(success=False, best_reward=-1.0) == 0.0


def test_difficulty_weights_are_simple_and_monotonic():
    weights = [
        difficulty_weight(
            EpisodeSpec(
                task="lift_cube", instruction="x", seed=1, difficulty=difficulty
            )
        )
        for difficulty in ["low", "medium", "hard", "very_high"]
    ]

    assert weights == sorted(weights)
    assert weights == [0.75, 1.0, 1.25, 1.5]


def test_lower_is_better_weighted_loss_matches_score_complement():
    scores = np.array([1.0, 0.4, 0.0], dtype=np.float32)
    weights = np.array([1.0, 2.0, 1.0], dtype=np.float32)

    weighted_score = compute_weighted_episode_score(scores, weights)
    loss = compute_weighted_loss(scores, weights)

    assert weighted_score == pytest.approx(0.45)
    assert loss == pytest.approx(0.55)


def test_robotics_metric_payload_separates_score_from_loss():
    invalid = RoboticsVLAValidationModule._invalid_metrics(
        RoboticsVLAValidationModule.__new__(RoboticsVLAValidationModule),
        "too many params",
    )

    assert invalid.score == 0.0
    assert invalid.loss == 1.0


def test_extract_frame_rotates_camera_180_degrees():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    frame = extract_frame({"agentview_image": image}, "agentview")

    np.testing.assert_array_equal(frame, np.rot90(image, 2))
    assert frame.flags.c_contiguous


def test_prepare_policy_obs_hides_raw_state_by_default():
    raw_obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
        "Can_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "object-state": np.ones(14, dtype=np.float32),
    }
    episode = EpisodeSpec(task="pick_place_can", instruction="pick can", seed=1)
    settings = RolloutSettings(
        seed=0,
        suite_version="panda_tabletop_v1",
        robot="Panda",
        controller="BASIC",
        horizon=320,
        max_episode_horizon=300,
        control_freq=20,
        camera_name="agentview",
        camera_height=2,
        camera_width=2,
        action_dim=7,
        render_video=False,
        video_dir="",
        max_videos=0,
    )

    obs = prepare_policy_obs(raw_obs, episode, settings, step_idx=3)

    assert set(obs) == {
        "image",
        "instruction",
        "proprio",
        "task",
        "step",
        "difficulty",
        "horizon",
    }
    assert obs["proprio"].shape == (25,)
    assert obs["difficulty"] is None
    assert obs["horizon"] == 320
    assert "raw_obs" not in obs
    assert "low_dim" not in obs


def test_prepare_policy_obs_uses_episode_camera_override():
    raw_obs = {
        "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "frontview_image": np.ones((2, 2, 3), dtype=np.uint8) * 255,
        "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        "robot0_joint_vel": np.zeros(7, dtype=np.float32),
        "robot0_eef_pos": np.zeros(3, dtype=np.float32),
        "robot0_eef_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "robot0_gripper_qpos": np.zeros(2, dtype=np.float32),
        "robot0_gripper_qvel": np.zeros(2, dtype=np.float32),
    }
    episode = EpisodeSpec(
        task="pick_place_can",
        instruction="pick can",
        seed=1,
        camera_name="frontview",
    )
    settings = RolloutSettings(
        seed=0,
        suite_version="panda_tabletop_v1",
        robot="Panda",
        controller="BASIC",
        horizon=320,
        max_episode_horizon=300,
        control_freq=20,
        camera_name="agentview",
        camera_height=2,
        camera_width=2,
        action_dim=7,
        render_video=False,
        video_dir="",
        max_videos=0,
    )

    obs = prepare_policy_obs(raw_obs, episode, settings, step_idx=0)

    assert obs["image"].mean() == 255


def test_domain_randomization_is_deterministic_and_manifest_level():
    manifest = ValidationManifest(
        episodes=[
            EpisodeSpec(
                episode_id="episode-1",
                task="pick_place_can",
                instruction="Pick up the can.",
                seed=100,
                horizon=120,
                camera_name="agentview",
                difficulty="low",
                tags=["public_eval"],
            ),
            EpisodeSpec(
                episode_id="episode-2",
                task="pick_place_milk",
                instruction="Pick up the milk.",
                seed=101,
                horizon=140,
                camera_name="agentview",
                difficulty="medium",
                tags=["public_eval"],
            ),
            EpisodeSpec(
                episode_id="episode-3",
                task="lift_cube",
                instruction="Lift the cube.",
                seed=102,
                horizon=90,
                camera_name="agentview",
                difficulty="low",
                tags=["public_eval"],
            ),
        ]
    )
    spec = {
        "version": DOMAIN_RANDOMIZATION_VERSION,
        "seed_salt": "unit-test",
        "rules": [
            {
                "name": "heldout_frontview_pick_place",
                "match": {"task": ["pick_place_can", "pick_place_milk"]},
                "camera_weights": {"frontview": 1.0},
                "horizon_jitter": {"min": 7, "max": 7},
                "seed_offset_range": {"min": 1000, "max": 1000},
                "instruction_prefixes": [
                    "Held-out visual variant: {instruction_lower}"
                ],
                "tags": ["heldout_visual_variant"],
                "modifiers": {"bin_texture": "matte_private_eval"},
            }
        ],
    }

    first = apply_domain_randomization_to_manifest(manifest, spec)
    second = apply_domain_randomization_to_manifest(manifest, spec)

    assert first.model_dump() == second.model_dump()
    randomized = [
        episode
        for episode in first.episodes
        if episode.task in {"pick_place_can", "pick_place_milk"}
    ]
    assert randomized
    for original, episode in zip(manifest.episodes, first.episodes):
        if episode.task in {"pick_place_can", "pick_place_milk"}:
            assert episode.camera_name == "frontview"
            assert episode.horizon == (original.horizon or 0) + 7
            assert episode.seed == original.seed + 1000
            assert episode.instruction.startswith("Held-out visual variant:")
            assert "heldout_visual_variant" in episode.tags
            assert (
                episode.domain_randomization["modifiers"]["bin_texture"]
                == "matte_private_eval"
            )


def test_validation_zip_package_applies_domain_randomization(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    randomization_path = tmp_path / "domain_randomization.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "episode_id": "episode-1",
                        "task": "pick_place_can",
                        "instruction": "Pick up the can.",
                        "seed": 7,
                        "horizon": 120,
                        "camera_name": "agentview",
                        "difficulty": "low",
                    }
                ],
            }
        )
    )
    randomization_path.write_text(
        json.dumps(
            {
                "version": DOMAIN_RANDOMIZATION_VERSION,
                "seed_salt": "zip-test",
                "rules": [
                    {
                        "name": "force_frontview",
                        "match": {"task": "pick_place_can"},
                        "camera_weights": {"frontview": 1.0},
                        "horizon_jitter": {"min": 5, "max": 5},
                        "tags": ["zip_randomized"],
                    }
                ],
            }
        )
    )
    write_validation_package(
        manifest_path, output_zip, randomization_path, metadata={"split": "public_eval"}
    )

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")
    episode = resolved.manifest.episodes[0]

    assert episode.camera_name == "frontview"
    assert episode.horizon == 125
    assert "zip_randomized" in episode.tags
    assert resolved.diagnostics["validation_data_source"] == "zip"
    assert resolved.diagnostics["domain_randomized_episodes"] == "1"


def test_validation_zip_package_loads_allowlisted_task_registry(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    registry_path = tmp_path / "task_registry.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [
                    {
                        "episode_id": "private-target-1",
                        "task": "targeted_clutter_pick_place",
                        "instruction": "Pick only the can into the matching bin.",
                        "seed": 9,
                        "horizon": 260,
                        "camera_name": "frontview",
                        "difficulty": "hard",
                        "task_kwargs": {"target_object": "can"},
                    }
                ],
            }
        )
    )
    registry_path.write_text(
        json.dumps(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "targeted_clutter_pick_place": {
                        "base_env": "PickPlace",
                        "env_kwargs": {"single_object_mode": 0},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": ["target_object"],
                        "partial_metrics": ["target_reach_score", "target_lift_score"],
                    }
                },
            }
        )
    )
    write_validation_package(
        manifest_path, output_zip, task_registry_path=registry_path
    )

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")
    episode = resolved.manifest.episodes[0]
    env_name, env_kwargs, task_entry = resolve_task_definition(
        episode,
        RolloutSettings(
            seed=0,
            suite_version="panda_tabletop_v1",
            robot="Panda",
            controller="BASIC",
            horizon=320,
            max_episode_horizon=300,
            control_freq=20,
            camera_name="agentview",
            camera_height=2,
            camera_width=2,
            action_dim=7,
            render_video=False,
            video_dir="",
            max_videos=0,
            task_registry=resolved.task_registry,
        ),
    )

    assert env_name == "PickPlace"
    assert env_kwargs == {"single_object_mode": 0}
    assert task_entry["success_checker"] == "target_object_in_matching_bin"
    assert resolved.diagnostics["task_registry_tasks"] == "targeted_clutter_pick_place"


def test_task_registry_rejects_non_allowlisted_code_or_env():
    with pytest.raises(ValueError, match="unsupported base_env"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "ArbitraryEnv",
                        "env_kwargs": {},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="unsupported success_checker"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "PickPlace",
                        "env_kwargs": {},
                        "success_checker": "__import__('os').system",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )
    with pytest.raises(ValueError, match="unsupported env_kwargs"):
        validate_task_registry_spec(
            {
                "version": TASK_REGISTRY_VERSION,
                "tasks": {
                    "unsafe": {
                        "base_env": "PickPlace",
                        "env_kwargs": {"robots": "Sawyer"},
                        "success_checker": "target_object_in_matching_bin",
                        "progress_checker": "target_pick_place_progress",
                        "allowed_task_kwargs": [],
                    }
                },
            }
        )


def test_targeted_clutter_success_uses_target_object_only():
    class Obj:
        def __init__(self, name):
            self.name = name

    class Env:
        objects = [Obj("Milk"), Obj("Bread"), Obj("Cereal"), Obj("Can")]
        object_to_id = {"milk": 0, "bread": 1, "cereal": 2, "can": 3}
        objects_in_bins = np.array([0, 0, 0, 1])

        def _check_success(self):
            return False

    assert target_object_in_matching_bin_success(
        Env(),
        EpisodeSpec(
            task="targeted_clutter_pick_place",
            instruction="pick can",
            seed=1,
            task_kwargs={"target_object": "can"},
        ),
    )
    assert not target_object_in_matching_bin_success(
        Env(),
        EpisodeSpec(
            task="targeted_clutter_pick_place",
            instruction="pick milk",
            seed=1,
            task_kwargs={"target_object": "milk"},
        ),
    )


def test_targeted_nut_success_uses_target_nut_only():
    class Env:
        nut_to_id = {"square": 0, "round": 1}
        objects_on_pegs = np.array([0, 1])

        def _check_success(self):
            return False

    assert target_nut_on_matching_peg_success(
        Env(),
        EpisodeSpec(
            task="distractor_nut_assembly",
            instruction="place round nut",
            seed=1,
            task_kwargs={"target_nut": "round"},
        ),
    )
    assert not target_nut_on_matching_peg_success(
        Env(),
        EpisodeSpec(
            task="distractor_nut_assembly",
            instruction="place square nut",
            seed=1,
            task_kwargs={"target_nut": "square"},
        ),
    )


def test_validation_set_url_alias_loads_robotics_zip(tmp_path: Path):
    manifest_path = tmp_path / "manifest.json"
    output_zip = tmp_path / "validation_package.zip"
    manifest_path.write_text(
        json.dumps(
            {
                "suite_version": "panda_tabletop_v1",
                "episodes": [{"task": "lift_cube", "instruction": "Lift.", "seed": 1}],
            }
        )
    )
    write_validation_package(manifest_path, output_zip)

    class Input:
        validation_data_url = None
        validation_zip_url = None
        validation_set_url = str(output_zip)
        validation_manifest_url = None
        domain_randomization_url = None

    resolved = resolve_validation_data_package(Input(), tmp_path / "cache")

    assert resolved.manifest.episodes[0].task == "lift_cube"
    assert resolved.diagnostics["validation_data_source"] == "zip"


def test_validation_zip_rejects_path_traversal(tmp_path: Path):
    output_zip = tmp_path / "bad.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr("../manifest.json", "{}")

    class Input:
        validation_data_url = str(output_zip)
        validation_zip_url = None
        validation_set_url = None
        validation_manifest_url = None
        domain_randomization_url = None

    with pytest.raises(ValueError, match="escapes"):
        resolve_validation_data_package(Input(), tmp_path / "cache")


def test_adapter_contract_loads_policy(tmp_path: Path):
    adapter_path = tmp_path / "flock_robotics_adapter.py"
    adapter_path.write_text(
        """
import numpy as np

class Policy:
    def act(self, obs):
        return np.zeros(7, dtype=np.float32)

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )

    assert policy.act({}).shape == (7,)


def test_adapter_worker_does_not_receive_validator_secrets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLOCK_API_KEY", "must-not-cross-process-boundary")
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-process-boundary")
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
import os
import numpy as np

class Policy:
    def act(self, obs):
        return np.zeros(7, dtype=np.float32)

def load_policy(model_dir, device, dtype):
    if os.getenv("FLOCK_API_KEY") or os.getenv("HF_TOKEN"):
        raise RuntimeError("validator secret leaked into adapter worker")
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )
    try:
        assert policy.act({}).shape == (7,)
    finally:
        policy.close()


def test_adapter_worker_enforces_action_wall_timeout(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
class Policy:
    def act(self, obs):
        while True:
            pass

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path,
        "flock_robotics_adapter.py",
        "cpu",
        "float32",
        action_timeout_seconds=0.1,
    )
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        policy.act({})
    assert excinfo.value.failure_mode == "policy_timeout"


def test_adapter_worker_counts_model_on_dependency_module_for_telemetry(
    tmp_path: Path,
):
    # Parameter count is telemetry (model size is enforced by the memory limit),
    # but it should still account for weights a policy parks on a dependency
    # module and reads back from act().
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
import numpy as np

np.HIDDEN = np.zeros(100, dtype=np.float32)

class Policy:
    def act(self, obs):
        return np.HIDDEN[:7]

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )
    try:
        assert policy.audited_parameter_count == 100
    finally:
        policy.close()


def test_adapter_worker_counts_object_wrapped_model_for_telemetry(tmp_path: Path):
    # A submitted-class wrapper nested inside a container on a dependency module
    # is still walked for the telemetry count.
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
import numpy as np

class _Weights:
    pass

_holder = _Weights()
_holder.model = np.zeros(100, dtype=np.float32)
np.HIDDEN = [_holder]

class Policy:
    def act(self, obs):
        return np.HIDDEN[0].model[:7]

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )
    try:
        assert policy.audited_parameter_count == 100
    finally:
        policy.close()


def test_adapter_worker_scan_of_dependency_module_does_not_inflate_count(
    tmp_path: Path,
):
    # Referencing a dependency module from act() (so the auditor scans it) must
    # not over-count a legitimately small policy.
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
import numpy as np

class Policy:
    def __init__(self):
        self.w = np.zeros(10, dtype=np.float32)

    def act(self, obs):
        return np.asarray(self.w[:7]) * np.float32(1.0)

def load_policy(model_dir, device, dtype):
    return Policy()
"""
    )

    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )
    try:
        assert policy.audited_parameter_count == 10
    finally:
        policy.close()


def test_policy_sandbox_rejects_over_budget_policy(tmp_path: Path):
    # The runtime memory ceiling is the authoritative model-size bound: a policy
    # that holds more than the budget is rejected regardless of how the weights
    # are represented. Here ~300 MB of resident bytes exceed a 200 MB limit.
    #
    # On macOS (and on the CUDA deployment) RLIMIT_AS is not the model-size cap,
    # so the host memory monitor observes the resident bytes and kills the worker
    # -> policy_memory_exceeded. On Linux/CPU the kernel RLIMIT_AS backstop trips
    # the allocation first -> model_load_failed. Both are the enforcement working.
    (tmp_path / "flock_robotics_adapter.py").write_text(
        """
import time

class Policy:
    def act(self, obs):
        return [0.0] * 7

def load_policy(model_dir, device, dtype):
    hog = bytearray(300_000_000)   # ~300 MB, resident (zero-filled by CPython)
    time.sleep(1.5)                # give the memory monitor time to observe it
    policy = Policy()
    policy._hog = hog
    return policy
"""
    )

    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(
            tmp_path,
            "flock_robotics_adapter.py",
            "cpu",
            "float32",
            memory_limit_bytes=200 * 1024 * 1024,
        )
    mode = excinfo.value.failure_mode
    if sys.platform == "darwin":
        assert mode == "policy_memory_exceeded"
    else:
        assert mode in {"policy_memory_exceeded", "model_load_failed"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux seccomp behavior")
def test_adapter_worker_denies_network_filesystem_and_process_escape(tmp_path: Path):
    import socket as host_socket

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    secret_path = tmp_path / "validator-secret.txt"
    secret_path.write_text("must-not-be-readable")
    # A live host daemon socket (stands in for e.g. /var/run/docker.sock). Landlock
    # does not mediate UNIX connect, so seccomp must refuse it.
    daemon_path = tmp_path / "host-daemon.sock"
    daemon = host_socket.socket(host_socket.AF_UNIX, host_socket.SOCK_STREAM)
    daemon.bind(str(daemon_path))
    daemon.listen(1)
    (model_dir / "flock_robotics_adapter.py").write_text(
        f"""
import os
import socket
import subprocess
import numpy as np

class Policy:
    def act(self, obs):
        return np.zeros(7, dtype=np.float32)

def _unix_connect():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect({str(daemon_path)!r})

def _unix_sendto():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    s.sendto(b"x", {str(daemon_path)!r})

def load_policy(model_dir, device, dtype):
    probes = [
        socket.socket,
        lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM),
        lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW),
        _unix_connect,
        _unix_sendto,
        socket.socketpair,
        # only the worker's own /proc entries are granted, never another process's
        lambda: open("/proc/%d/cmdline" % os.getppid()).read(),
        lambda: open({str(secret_path)!r}).read(),
        lambda: subprocess.run(["/bin/true"], check=True),
    ]
    denied = 0
    for probe in probes:
        try:
            probe()
        except OSError:
            denied += 1
    # allowed: a bare local socket (the CUDA driver creates one)
    socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET).close()
    if denied == len(probes):
        return Policy()
    raise RuntimeError("one or more sandbox boundaries were not enforced")
"""
    )

    try:
        policy = load_policy_from_adapter(
            model_dir, "flock_robotics_adapter.py", "cpu", "float32"
        )
        try:
            assert policy.act({}).shape == (7,)
        finally:
            policy.close()
    finally:
        daemon.close()


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(
    sys.platform != "linux" or not _cuda_available(), reason="needs Linux with a CUDA GPU"
)
def test_adapter_worker_runs_cuda_inside_the_sandbox_within_the_memory_cap(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "flock_robotics_adapter.py").write_text(
        """
import numpy as np
import torch

class Policy:
    def __init__(self, device):
        self.device = device
        self.net = torch.nn.Linear(7, 7).to(device)

    def act(self, obs):
        x = torch.zeros(1, 7, device=self.device)
        return self.net(x).detach().cpu().numpy()[0].clip(-1, 1)

def load_policy(model_dir, device, dtype):
    return Policy(device)
"""
    )
    policy = load_policy_from_adapter(
        model_dir,
        "flock_robotics_adapter.py",
        "cuda",
        "bfloat16",
        memory_limit_bytes=4 * 1024**3,
    )
    try:
        assert policy.act({}).shape == (7,)
    finally:
        policy.close()

    oversize = tmp_path / "oversize"
    oversize.mkdir()
    (oversize / "flock_robotics_adapter.py").write_text(
        """
import numpy as np
import torch

class Policy:
    def act(self, obs):
        return np.zeros(7, dtype=np.float32)

def load_policy(model_dir, device, dtype):
    policy = Policy()
    policy.big = torch.empty(int(8 * 1024**3), dtype=torch.uint8, device=device)
    return policy
"""
    )
    with pytest.raises(RoboticsSubmissionError):
        load_policy_from_adapter(
            oversize,
            "flock_robotics_adapter.py",
            "cuda",
            "bfloat16",
            memory_limit_bytes=4 * 1024**3,
        )


# --- Issue 1: bad submissions are scored, never crash the validator ----------


def test_adapter_loads_through_hf_style_symlink(tmp_path: Path):
    """HF snapshots symlink repo files into a sibling blobs/ dir; the adapter
    loader must follow that without flagging it as a path escape."""
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "deadbeef"
    blob.write_text(
        "import numpy as np\n"
        "class Policy:\n"
        "    def act(self, obs):\n"
        "        return np.zeros(7, dtype=np.float32)\n"
        "def load_policy(model_dir, device, dtype):\n"
        "    return Policy()\n"
    )
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "flock_robotics_adapter.py").symlink_to(blob)

    policy = load_policy_from_adapter(
        snapshot, "flock_robotics_adapter.py", "cpu", "float32"
    )

    assert policy.act({}).shape == (7,)


def test_adapter_filename_escape_is_rejected(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text(
        "def load_policy(model_dir, device, dtype):\n    return object()\n"
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(
            model_dir, "../flock_robotics_adapter.py", "cpu", "float32"
        )
    assert excinfo.value.failure_mode == "adapter_contract"


def test_missing_adapter_is_submission_error(tmp_path: Path):
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(
            tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
        )
    assert excinfo.value.failure_mode == "adapter_missing"


def test_adapter_without_load_policy_is_contract_error(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text("X = 1\n")
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(
            tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
        )
    assert excinfo.value.failure_mode == "adapter_contract"


def test_policy_without_act_is_contract_error(tmp_path: Path):
    (tmp_path / "flock_robotics_adapter.py").write_text(
        "def load_policy(model_dir, device, dtype):\n    return object()\n"
    )
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        load_policy_from_adapter(
            tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
        )
    assert excinfo.value.failure_mode == "adapter_contract"


def test_failing_load_policy_is_retried_then_succeeds(tmp_path: Path):
    from validator.modules.robotics_vla.adapter import MODEL_QUERY_RETRIES

    (tmp_path / "flock_robotics_adapter.py").write_text(
        f"""
import numpy as np

attempts = 0

class Policy:
    def act(self, obs):
        result = np.zeros(7, dtype=np.float32)
        result[0] = attempts
        return result

def load_policy(model_dir, device, dtype):
    global attempts
    attempts += 1
    if attempts <= {MODEL_QUERY_RETRIES}:
        raise RuntimeError("transient model load failure")
    return Policy()
"""
    )
    policy = load_policy_from_adapter(
        tmp_path, "flock_robotics_adapter.py", "cpu", "float32"
    )
    try:
        assert policy.act({})[0] == 1 + MODEL_QUERY_RETRIES
    finally:
        policy.close()


def test_query_policy_action_retries_then_reports_invalid_action():
    from validator.modules.robotics_vla.adapter import MODEL_QUERY_RETRIES

    class BadShapePolicy:
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            return np.zeros(6, dtype=np.float32)

    policy = BadShapePolicy()
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        query_policy_action(policy, {}, 7)
    assert excinfo.value.failure_mode == "invalid_action"
    assert policy.calls == 1 + MODEL_QUERY_RETRIES


def test_query_policy_action_recovers_from_transient_failure():
    class FlakyPolicy:
        def __init__(self):
            self.calls = 0

        def act(self, obs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("transient cuda error")
            return np.zeros(7, dtype=np.float32)

    policy = FlakyPolicy()
    action = query_policy_action(policy, {}, 7)
    assert action.shape == (7,)
    assert policy.calls == 3


def test_invalid_metrics_carry_failure_mode():
    metrics = RoboticsVLAValidationModule._invalid_metrics(
        RoboticsVLAValidationModule.__new__(RoboticsVLAValidationModule),
        "broken adapter",
        failure_mode="adapter_missing",
    )
    assert metrics.invalid_submission is True
    assert metrics.score == 0.0
    assert metrics.diagnostics["failure_mode"] == "adapter_missing"


def test_runner_does_not_crash_on_bad_submission():
    """A submission that raises must fail just that assignment, not exit(1)."""
    import sys as _sys

    from validator.validation_runner import ValidationRunner

    class BoomModule:
        def __init__(self):
            self.calls = 0

        def validate(self, data):
            self.calls += 1
            raise ValueError("policy action must have shape (7,)")

    class FakeApi:
        def __init__(self):
            self.failed = []

        def mark_assignment_as_failed(self, assignment_id):
            self.failed.append(assignment_id)

    runner = ValidationRunner.__new__(ValidationRunner)
    module = BoomModule()
    runner.task_id_to_module = {"task-1": module}
    runner.api = FakeApi()

    real_exit = _sys.exit
    _sys.exit = lambda *a: (_ for _ in ()).throw(
        AssertionError("runner called sys.exit")
    )
    try:
        result = runner.perform_validation("assignment-1", "task-1", object())
    finally:
        _sys.exit = real_exit

    assert result is None
    assert module.calls == 3
    assert runner.api.failed == ["assignment-1"]


# --- Parameter counting is telemetry; model size is enforced by memory --------


def test_count_policy_parameters_handles_non_torch_policy():
    class Policy:
        def act(self, obs):
            return None

    # An inspectable policy with no tensor state has an auditable zero count.
    assert count_policy_parameters(Policy()) == 0


def test_count_policy_parameters_recurses_into_nested_containers_and_cycles(
    monkeypatch,
):
    class FakeTensor:
        def __init__(self, size):
            self._size = size

        def numel(self):
            return self._size

    class FakeModule:
        pass

    class FakeLinear(FakeModule):
        def __init__(self):
            self.weight = FakeTensor(6)
            self.bias = FakeTensor(2)

    fake_torch = SimpleNamespace(
        Tensor=FakeTensor,
        nn=SimpleNamespace(Module=FakeModule),
        device=type("FakeDevice", (), {}),
        dtype=type("FakeDtype", (), {}),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Policy:
        def __init__(self):
            nested = []
            nested.append(nested)
            nested.append({"model": FakeLinear()})
            self.payload = nested

        def act(self, obs):
            return np.zeros(7, dtype=np.float32)

    # Linear(3, 2) has 6 weights and 2 bias parameters.
    assert count_policy_parameters(Policy()) == 8


def test_count_policy_parameters_rejects_opaque_policy_state():
    class Opaque:
        __slots__ = ()

    class Policy:
        def __init__(self):
            self.audited_parameter_count = 0  # Miner self-attestation is not trusted.
            self.hidden = Opaque()

        def act(self, obs):
            return None

    assert count_policy_parameters(Policy()) is None


def test_count_policy_parameters_counts_class_level_model():
    # A miner cannot bypass the cap by stashing the model as a class attribute
    # instead of an instance attribute.
    big = np.zeros(10_000_000, dtype=np.float32)

    class ClassAttrPolicy:
        weights = {"w": big}

        def act(self, obs):
            return type(self).weights["w"][:7]

    assert count_policy_parameters(ClassAttrPolicy()) == 10_000_000


def test_count_policy_parameters_counts_property_hidden_model():
    # Nor by hiding the model behind a property getter.
    big = np.zeros(10_000_000, dtype=np.float32)

    class PropertyPolicy:
        @property
        def weights(self):
            return big

        def act(self, obs):
            return self.weights[:7]

    assert count_policy_parameters(PropertyPolicy()) == 10_000_000


def test_invalid_hub_reference_is_classified_as_submission_error(monkeypatch):
    response = SimpleNamespace(status_code=404, headers={}, request=None)
    error = hf_errors.HfHubHTTPError("missing repo", response=response)

    def fail_download(**_kwargs):
        raise error

    monkeypatch.setattr(
        "validator.modules.robotics_vla.adapter.snapshot_download",
        fail_download,
    )
    with pytest.raises(RoboticsSubmissionError) as excinfo:
        resolve_model_dir("missing/repo", "bad-revision")
    assert excinfo.value.failure_mode == "model_reference_invalid"


def test_invalid_hub_reference_is_returned_as_zero_score_metrics(monkeypatch):
    def fail_resolution(*_args, **_kwargs):
        raise RoboticsSubmissionError(
            "missing model revision",
            failure_mode="model_reference_invalid",
        )

    monkeypatch.setattr(
        "validator.modules.robotics_vla.resolve_model_dir",
        fail_resolution,
    )
    module = RoboticsVLAValidationModule(config=RoboticsVLAConfig(device="cpu"))
    metrics = module.validate(RoboticsVLAInputData(hg_repo_id="missing/repo"))

    assert metrics.invalid_submission is True
    assert metrics.score == 0.0
    assert metrics.loss == 1.0
    assert metrics.diagnostics["failure_mode"] == "model_reference_invalid"


def test_transient_hub_failure_remains_recoverable(monkeypatch):
    response = SimpleNamespace(status_code=503, headers={}, request=None)
    error = hf_errors.HfHubHTTPError("service unavailable", response=response)

    def fail_download(**_kwargs):
        raise error

    monkeypatch.setattr(
        "validator.modules.robotics_vla.adapter.snapshot_download",
        fail_download,
    )
    with pytest.raises(hf_errors.HfHubHTTPError):
        resolve_model_dir("org/repo", "main")


def test_offline_cache_miss_remains_recoverable(monkeypatch):
    error = hf_errors.LocalEntryNotFoundError("not cached while offline")

    def fail_download(**_kwargs):
        raise error

    monkeypatch.setattr(
        "validator.modules.robotics_vla.adapter.snapshot_download",
        fail_download,
    )
    with pytest.raises(hf_errors.LocalEntryNotFoundError):
        resolve_model_dir("org/repo", "main")


# --- Runtime memory ceiling (the authoritative model-size enforcement) --------


def test_read_process_rss_reports_plausible_value_for_self():
    import os

    from validator.modules.robotics_vla.memory_monitor import read_process_rss_bytes

    rss = read_process_rss_bytes(os.getpid())
    assert isinstance(rss, int) and rss > 1_000_000


def test_memory_monitor_kills_process_over_limit():
    import subprocess

    from validator.modules.robotics_vla.memory_monitor import MemoryMonitor

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    monitor = MemoryMonitor(
        proc.pid,
        limit_bytes=1_000,
        sampler=lambda _pid: 2_000,  # force a breach without a huge allocation
        poll_interval_seconds=0.01,
        breaches_before_kill=2,
    )
    monitor.start()
    try:
        proc.wait(timeout=5)
    finally:
        monitor.stop()
    assert monitor.breached is True
    assert monitor.observed_at_kill == 2_000
    assert proc.poll() is not None


def test_memory_monitor_leaves_process_under_limit_alone():
    import subprocess
    import time

    from validator.modules.robotics_vla.memory_monitor import MemoryMonitor

    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"], start_new_session=True
    )
    monitor = MemoryMonitor(
        proc.pid,
        limit_bytes=10**12,
        sampler=lambda _pid: 5_000,
        poll_interval_seconds=0.01,
        breaches_before_kill=2,
    )
    monitor.start()
    try:
        time.sleep(0.2)
        assert proc.poll() is None  # still alive while under budget
    finally:
        monitor.stop()
        proc.terminate()
        proc.wait(timeout=5)
    assert monitor.breached is False


def test_memory_monitor_is_disabled_for_nonpositive_limit():
    from validator.modules.robotics_vla.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(
        pid=-1, limit_bytes=0, sampler=lambda _pid: 10**9, poll_interval_seconds=0.01
    )
    monitor.start()  # no-op: an unbounded limit means no monitoring thread
    monitor.stop()
    assert monitor.breached is False
