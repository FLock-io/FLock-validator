"""Tests for the video inconsistency validation module (schemas + validate flow).

The package, model resolution, sandboxed detector and video decoding are replaced
in the module namespace so these tests exercise only the module's own logic.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from validator.config import load_config_for_task
from validator.exceptions import RecoverableException
from validator.modules.video_inconsistency import module as vi_module
from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.issue_types import SUITE_VERSION
from validator.modules.video_inconsistency.manifest import (
    ClipSpec,
    DecoyLabel,
    IssueLabel,
    VideoManifest,
)
from validator.modules.video_inconsistency.module import (
    VideoInconsistencyConfig,
    VideoInconsistencyInputData,
    VideoInconsistencyMetrics,
    VideoInconsistencyValidationModule,
)


FPS = 10.0
FRAMES = 50  # 5 s clips
HEIGHT, WIDTH = 8, 8


# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


def make_clip(
    index: int,
    edited: bool,
    difficulty: str = "medium",
    decoys: list[DecoyLabel] | None = None,
) -> ClipSpec:
    issues = []
    if edited:
        issues = [
            IssueLabel(
                type="frozen_frames",
                start_time=1.0,
                end_time=2.0,
                start_frame=10,
                end_frame=20,
            )
        ]
    return ClipSpec(
        clip_id=f"clip{index:02d}",
        video_path=f"videos/clip{index:02d}.mp4",
        fps=FPS,
        num_frames=FRAMES,
        width=WIDTH,
        height=HEIGHT,
        difficulty=difficulty,
        issues=issues,
        decoys=decoys or [],
    )


def make_manifest(num_clips: int = 8, suite_version: str = SUITE_VERSION) -> VideoManifest:
    # Last quarter of the clips are clean.
    clean_from = num_clips - max(1, num_clips // 4)
    return VideoManifest(
        suite_version=suite_version,
        clips=[make_clip(i, edited=i < clean_from) for i in range(num_clips)],
    )


class FakePackage:
    def __init__(self, manifest: VideoManifest, root: Path):
        self.manifest = manifest
        self.root = root
        self.diagnostics = {"package_source": "fake"}

    def clip_video_path(self, clip: ClipSpec) -> Path:
        return self.root / clip.video_path


class FakeDetector:
    """Scriptable stand-in for IsolatedDetector; answers keyed by clip id (video stem)."""

    def __init__(self, manifest: VideoManifest, behaviour=None):
        self.labels = {c.clip_id: c.issues for c in manifest.clips}
        self.behaviour = behaviour
        self.calls: dict[str, int] = {}
        self.closed = 0
        self.parameter_count = 123_456
        self.detect_kwargs: list[set[str]] = []

    def perfect_answer(self, clip_id: str) -> dict:
        return {
            "issues": [
                {
                    "type": issue.type,
                    "start_time": issue.start_time,
                    "end_time": issue.end_time,
                    "confidence": 0.9,
                    "bbox": issue.bbox,
                }
                for issue in self.labels[clip_id]
            ]
        }

    def detect(self, *, frames, video_path, fps):
        clip_id = Path(video_path).stem
        self.detect_kwargs.append({"frames", "video_path", "fps"})
        self.calls[clip_id] = self.calls.get(clip_id, 0) + 1
        assert frames.shape == (FRAMES, HEIGHT, WIDTH, 3)
        if self.behaviour is not None:
            outcome = self.behaviour(self, clip_id, self.calls[clip_id])
            if outcome is not None:
                return outcome
        return self.perfect_answer(clip_id)

    def close(self):
        self.closed += 1


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """Patch the module's collaborators; returns a namespace to tweak per test."""

    class Harness:
        pass

    h = Harness()
    h.manifest = make_manifest()
    h.package = FakePackage(h.manifest, tmp_path)
    h.detector = FakeDetector(h.manifest)
    h.load_kwargs = {}
    h.calls = {"package": 0, "model_dir": 0, "load": 0}

    def fake_resolve_package(url, cache_dir):
        h.calls["package"] += 1
        h.package_args = (url, cache_dir)
        return h.package

    def fake_resolve_model_dir(repo_id, revision="main", *, allow_local=False):
        h.calls["model_dir"] += 1
        h.model_args = (repo_id, revision)
        h.allow_local = allow_local
        return tmp_path / "model"

    def fake_load(model_dir, adapter_filename, **kwargs):
        h.calls["load"] += 1
        h.load_kwargs = {"model_dir": model_dir, "adapter_filename": adapter_filename, **kwargs}
        return h.detector

    def fake_decode(path, *, max_frames=None):
        n = FRAMES if h.decode_frames is None else h.decode_frames
        return np.zeros((n, HEIGHT, WIDTH, 3), dtype=np.uint8)

    h.decode_frames = None
    monkeypatch.setattr(vi_module, "resolve_validation_package", fake_resolve_package)
    monkeypatch.setattr(vi_module, "resolve_model_dir", fake_resolve_model_dir)
    monkeypatch.setattr(vi_module, "load_detector_from_adapter", fake_load)
    monkeypatch.setattr(vi_module, "decode_video", fake_decode)
    return h


def make_module(**config_overrides) -> VideoInconsistencyValidationModule:
    config = VideoInconsistencyConfig(device="cpu", **config_overrides)
    return VideoInconsistencyValidationModule(config=config)


def make_input(**overrides) -> VideoInconsistencyInputData:
    values = {"hg_repo_id": "org/detector", "validation_data_url": "/tmp/pkg.zip"}
    values.update(overrides)
    return VideoInconsistencyInputData(**values)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def test_config_defaults_match_production_config_file():
    from_file = load_config_for_task(
        task_id="anything",
        task_type="video_inconsistency",
        config_model=VideoInconsistencyConfig,
        config_dir="configs",
    )
    assert from_file == VideoInconsistencyConfig()
    assert from_file.device == "cuda"
    assert from_file.detector_memory_limit_gb == 18
    assert from_file.suite_version == SUITE_VERSION


def test_config_builds_scoring_settings():
    settings = VideoInconsistencyConfig(tiou_threshold=0.4, max_predictions_per_clip=5).scoring_settings()
    assert settings.tiou_threshold == 0.4
    assert settings.max_predictions_per_clip == 5
    assert dict(settings.difficulty_weights) == {
        "easy": 1.0,
        "medium": 1.5,
        "hard": 2.0,
        "expert": 2.5,
    }
    assert settings.tiou_thresholds == (0.3, 0.4, 0.5, 0.6, 0.7)
    assert settings.bbox_iou_threshold == 0.3
    assert (settings.weight_map, settings.weight_f1) == (0.75, 0.0)
    assert (settings.weight_localization, settings.weight_clip_accuracy) == (0.2, 0.05)


def test_config_passes_new_scoring_fields_through():
    settings = VideoInconsistencyConfig(
        tiou_thresholds=[0.5, 0.75],
        bbox_iou_threshold=0.5,
        weight_map=0.5,
        weight_f1=0.25,
        difficulty_weights={"easy": 1.0, "medium": 1.0, "hard": 1.0, "expert": 4.0},
    ).scoring_settings()
    assert settings.tiou_thresholds == (0.5, 0.75)
    assert settings.bbox_iou_threshold == 0.5
    assert settings.weight_map == 0.5 and settings.weight_f1 == 0.25
    assert settings.weight_for("expert") == 4.0


def test_config_allows_zero_score_term_weights():
    # v1-style scoring is still expressible: mAP off, F1 on.
    config = VideoInconsistencyConfig(
        weight_map=0.0, weight_f1=0.6, weight_localization=0.25, weight_clip_accuracy=0.15
    )
    assert config.scoring_settings().weight_map == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"weight_f1": 0.9},  # weights no longer sum to 1
        {"weight_map": 0.5},  # sums to 0.75
        {"weight_map": 1.0},  # sums to 1.25
        {"tiou_thresholds": []},
        {"tiou_thresholds": [0.0, 0.5]},
        {"tiou_thresholds": [0.5, 0.5]},
        {"bbox_iou_threshold": 0.0},
        {"bbox_iou_threshold": 1.5},
        {"tiou_threshold": 0.0},
        {"tiou_threshold": 1.5},
        {"confidence_threshold": 2.0},
        {"max_predictions_per_clip": 0},
        {"detector_memory_limit_gb": 0},
        {"detect_timeout_seconds": 0},
        {"detect_retries": -1},
        {"max_failed_clip_fraction": 1.5},
        {"max_clips": 0},
        {"difficulty_weights": {"easy": 1.0}},
        # The expert tier must have a weight.
        {"difficulty_weights": {"easy": 1.0, "medium": 1.5, "hard": 2.0}},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValidationError):
        VideoInconsistencyConfig(**overrides)


def test_input_data_defaults_and_aliases():
    data = VideoInconsistencyInputData(hg_repo_id="org/model")
    assert data.revision == "main"
    assert data.adapter_filename == "flock_video_adapter.py"
    for field in ("validation_data_url", "validation_zip_url", "validation_set_url"):
        parsed = VideoInconsistencyInputData.model_validate({"hg_repo_id": "org/m", field: "https://x/p.zip"})
        assert getattr(parsed, field) == "https://x/p.zip"


def test_fedledger_payload_parses_into_input_schema():
    resp = {
        "id": "assignment-1",
        "task_submission": {
            "data": {"hg_repo_id": "org/vic-detector", "revision": "abc", "submitter": "0xminer", "round": 3}
        },
        "data": {"validation_data_url": "https://fed-ledger.example/private_video_package.zip"},
    }
    merged = {**resp["task_submission"]["data"], **resp["data"]}
    data = VideoInconsistencyInputData.model_validate(merged)
    assert data.hg_repo_id == "org/vic-detector"
    assert data.revision == "abc"
    assert data.validation_data_url.endswith(".zip")
    assert data.adapter_filename == "flock_video_adapter.py"


def test_fedledger_rejects_payload_without_repo():
    with pytest.raises(ValidationError):
        VideoInconsistencyInputData.model_validate({"validation_data_url": "https://x/p.zip"})


def test_metrics_serialise_for_submission():
    metrics = VideoInconsistencyMetrics(
        score=0.5,
        loss=0.5,
        macro_f1=0.5,
        micro_f1=0.5,
        precision=0.5,
        recall=0.5,
        localization_score=0.5,
        clip_accuracy=0.5,
        mean_ap=0.5,
        per_type_ap={"zoom_jump": 0.5},
        ap_by_tiou={"0.3": 0.6, "0.5": 0.4},
        per_type_f1={"zoom_jump": 0.5},
        clips_evaluated=4,
        clips_failed=1,
        parameter_count=10,
        diagnostics={"k": "v"},
    )
    dump = metrics.model_dump()
    json.dumps({"status": "completed", "data": dump})
    assert dump["invalid_submission"] is False
    assert dump["mean_ap"] == 0.5
    assert dump["per_type_ap"] == {"zoom_jump": 0.5}
    assert dump["ap_by_tiou"] == {"0.3": 0.6, "0.5": 0.4}


def test_invalid_metrics_shape():
    module = VideoInconsistencyValidationModule.__new__(VideoInconsistencyValidationModule)
    metrics = module._invalid_metrics("bad", parameter_count=7, failure_mode="adapter_missing")
    dump = metrics.model_dump()
    json.dumps({"status": "completed", "data": dump})
    assert dump["invalid_submission"] is True
    assert dump["score"] == 0.0 and dump["loss"] == 1.0
    assert dump["mean_ap"] == 0.0
    assert dump["per_type_ap"] == {} and dump["ap_by_tiou"] == {}
    assert dump["parameter_count"] == 7
    assert dump["diagnostics"] == {"reason": "bad", "failure_mode": "adapter_missing"}


# --------------------------------------------------------------------------
# validate()
# --------------------------------------------------------------------------


def test_perfect_detector_scores_one(harness):
    metrics = make_module().validate(make_input())
    assert metrics.invalid_submission is False
    assert metrics.score == pytest.approx(1.0)
    assert metrics.loss == pytest.approx(0.0)
    assert metrics.macro_f1 == pytest.approx(1.0)
    assert metrics.mean_ap == pytest.approx(1.0)
    assert metrics.per_type_ap == {"frozen_frames": pytest.approx(1.0)}
    assert metrics.ap_by_tiou == {
        key: pytest.approx(1.0) for key in ("0.3", "0.4", "0.5", "0.6", "0.7")
    }
    assert metrics.clip_accuracy == pytest.approx(1.0)
    assert metrics.clips_evaluated == 8
    assert metrics.clips_failed == 0
    assert metrics.parameter_count == 123_456
    assert metrics.per_type_f1 == {"frozen_frames": pytest.approx(1.0)}
    assert harness.detector.closed == 1
    assert all(count == 1 for count in harness.detector.calls.values())
    # Diagnostics: package diagnostics, failure counts, per-type counts, timing.
    diagnostics = metrics.diagnostics
    assert diagnostics["package_source"] == "fake"
    assert json.loads(diagnostics["failure_counts"]) == {}
    assert json.loads(diagnostics["per_type_counts"])["frozen_frames"]["tp"] > 0
    assert "detect_seconds" in diagnostics and diagnostics["parameter_count"] == "123456"
    assert json.loads(diagnostics["decoy_counts"]) == {}
    json.dumps(metrics.model_dump())


def test_decoy_counts_are_reported_in_diagnostics(harness):
    decoys = [
        DecoyLabel(type="scene_cut", start_time=0.5, end_time=0.5),
        DecoyLabel(type="smooth_zoom", start_time=3.0, end_time=4.0),
    ]
    clips = [
        make_clip(0, edited=True, difficulty="expert", decoys=decoys),
        make_clip(1, edited=True, decoys=decoys[:1]),
        make_clip(2, edited=False, decoys=decoys[:1]),
    ]
    harness.manifest = VideoManifest(clips=clips)
    harness.package = FakePackage(harness.manifest, harness.package.root)
    harness.detector = FakeDetector(harness.manifest)
    metrics = make_module().validate(make_input())
    assert json.loads(metrics.diagnostics["decoy_counts"]) == {"scene_cut": 3, "smooth_zoom": 1}
    # Decoys are telemetry only: a perfect detector still scores 1.
    assert metrics.score == pytest.approx(1.0)
    assert metrics.mean_ap == pytest.approx(1.0)

    limited = make_module(max_clips=1).validate(make_input())
    assert json.loads(limited.diagnostics["decoy_counts"]) == {"scene_cut": 1, "smooth_zoom": 1}


def test_passes_config_and_inputs_through(harness):
    module = make_module(
        detector_memory_limit_gb=3,
        detector_load_timeout_seconds=11,
        detect_timeout_seconds=7,
        detector_cpu_time_seconds=99,
        torch_dtype="float16",
    )
    module.validate(make_input(revision="rev1", adapter_filename="my_adapter.py"))
    assert harness.package_args == ("/tmp/pkg.zip", module.config.package_cache_dir)
    assert harness.model_args == ("org/detector", "rev1")
    kwargs = harness.load_kwargs
    assert kwargs["adapter_filename"] == "my_adapter.py"
    assert kwargs["device"] == "cpu"
    assert kwargs["torch_dtype"] == "float16"
    assert kwargs["memory_limit_bytes"] == 3 * 1024**3
    assert kwargs["load_timeout_seconds"] == 11
    assert kwargs["detect_timeout_seconds"] == 7
    # The configured CPU budget is a floor, scaled to the run's worst-case wall
    # time on every core so RLIMIT_CPU can never fire before a wall-time limit.
    planned = len(harness.package.manifest.clips)
    worst_wall = 11 + planned * (module.config.detect_retries + 1) * 7
    assert kwargs["cpu_time_seconds"] == max(99, math.ceil(worst_wall * (os.cpu_count() or 1)))
    # Production never resolves a trainer reference to a host path, and the worker
    # is denied the extracted package (hidden labels) and the validator's .env.
    assert harness.allow_local is False
    assert harness.package.root in kwargs["deny_read_paths"]
    assert any(p.name == ".env" for p in kwargs["deny_read_paths"])


def test_cpu_budget_never_drops_below_the_configured_floor(harness):
    module = make_module(detector_cpu_time_seconds=10**9)
    assert module._cpu_time_budget(3) == 10**9
    assert make_module()._cpu_time_budget(500) >= 500 * 3 * 60


def test_package_url_alias_is_used(harness):
    make_module().validate(
        VideoInconsistencyInputData(hg_repo_id="org/m", validation_set_url="https://x/set.zip")
    )
    assert harness.package_args[0] == "https://x/set.zip"


def test_missing_package_url_is_an_infra_error(harness):
    with pytest.raises(ValueError, match="validation"):
        make_module().validate(VideoInconsistencyInputData(hg_repo_id="org/m"))
    assert harness.calls["model_dir"] == 0 and harness.calls["load"] == 0


def test_empty_detector_scores_poorly(harness):
    harness.detector.behaviour = lambda d, cid, n: {"issues": []}
    metrics = make_module().validate(make_input())
    assert metrics.invalid_submission is False
    assert metrics.macro_f1 == 0.0
    assert metrics.mean_ap == 0.0
    assert metrics.score < 0.1
    assert metrics.clips_failed == 0


def test_hallucinating_detector_is_penalised(harness):
    def behaviour(d, cid, n):
        answer = d.perfect_answer(cid)
        answer["issues"].append(
            {"type": "zoom_jump", "start_time": 0.0, "end_time": 1.0, "confidence": 0.9}
        )
        return answer

    harness.detector.behaviour = behaviour
    metrics = make_module().validate(make_input())
    assert metrics.score < 0.9
    assert metrics.per_type_f1["zoom_jump"] == 0.0
    assert metrics.per_type_ap["zoom_jump"] == 0.0
    assert metrics.mean_ap == pytest.approx(0.5)  # frozen_frames 1, zoom_jump 0


def test_non_fatal_error_is_retried_then_succeeds(harness):
    def behaviour(d, cid, attempt):
        if cid == "clip00" and attempt < 3:
            raise VideoSubmissionError("boom", "detector_execution_failed", fatal=False)

    harness.detector.behaviour = behaviour
    metrics = make_module(detect_retries=2).validate(make_input())
    assert harness.detector.calls["clip00"] == 3
    assert metrics.clips_failed == 0
    assert metrics.score == pytest.approx(1.0)


def test_invalid_output_is_retried_like_an_execution_error(harness):
    def behaviour(d, cid, attempt):
        if cid == "clip01" and attempt == 1:
            return {"issues": [{"type": "not_a_type", "start_time": 0, "end_time": 1}]}

    harness.detector.behaviour = behaviour
    metrics = make_module(detect_retries=1).validate(make_input())
    assert harness.detector.calls["clip01"] == 2
    assert metrics.clips_failed == 0
    assert metrics.score == pytest.approx(1.0)


def test_persistent_failures_below_threshold_are_scored_as_empty(harness):
    def behaviour(d, cid, attempt):
        if cid == "clip00":
            raise VideoSubmissionError("boom", "detector_execution_failed", fatal=False)

    harness.detector.behaviour = behaviour
    # 8 clips, 1 failure = 12.5% <= 25%.
    metrics = make_module(detect_retries=2).validate(make_input())
    assert metrics.invalid_submission is False
    assert harness.detector.calls["clip00"] == 3
    assert metrics.clips_failed == 1
    assert metrics.clips_evaluated == 8
    assert json.loads(metrics.diagnostics["failure_counts"]) == {"detector_execution_failed": 1}
    # The failed edited clip is a miss, so the score drops but stays positive.
    assert 0.0 < metrics.score < 1.0
    assert harness.detector.closed == 1


def test_persistent_failures_above_threshold_invalidate_and_abort_early(harness):
    def behaviour(d, cid, attempt):
        # clip00..clip02 fail with two different modes; 3 of 8 = 37.5% > 25%.
        if cid in ("clip00", "clip01"):
            raise VideoSubmissionError("boom", "detector_execution_failed", fatal=False)
        if cid == "clip02":
            return "garbage"  # -> detector_output_invalid

    harness.detector.behaviour = behaviour
    metrics = make_module(detect_retries=0).validate(make_input())
    assert metrics.invalid_submission is True
    assert metrics.score == 0.0 and metrics.loss == 1.0
    assert metrics.diagnostics["failure_mode"] == "detector_execution_failed"
    # Aborted at the 3rd failure: later clips never reached the detector.
    assert "clip03" not in harness.detector.calls
    assert harness.detector.closed == 1


def test_zero_tolerance_invalidates_on_first_persistent_failure(harness):
    def behaviour(d, cid, attempt):
        if cid == "clip00":
            raise VideoSubmissionError("boom", "detector_execution_failed", fatal=False)

    harness.detector.behaviour = behaviour
    metrics = make_module(detect_retries=0, max_failed_clip_fraction=0.0).validate(make_input())
    assert metrics.invalid_submission is True
    assert metrics.diagnostics["failure_mode"] == "detector_execution_failed"


def test_fatal_detect_error_invalidates_with_its_failure_mode(harness):
    def behaviour(d, cid, attempt):
        if cid == "clip02":
            raise VideoSubmissionError("hung", "detector_timeout", fatal=True)

    harness.detector.behaviour = behaviour
    metrics = make_module().validate(make_input())
    assert metrics.invalid_submission is True
    assert metrics.diagnostics["failure_mode"] == "detector_timeout"
    assert metrics.parameter_count == 123_456
    assert harness.detector.calls["clip02"] == 1  # fatal errors are not retried
    assert "clip03" not in harness.detector.calls
    assert harness.detector.closed == 1


def test_load_failure_invalidates_without_a_detector(harness, monkeypatch):
    def failing_load(*args, **kwargs):
        raise VideoSubmissionError("cannot import", "adapter_import_failed")

    monkeypatch.setattr(vi_module, "load_detector_from_adapter", failing_load)
    metrics = make_module().validate(make_input())
    assert metrics.invalid_submission is True
    assert metrics.diagnostics["failure_mode"] == "adapter_import_failed"
    assert harness.detector.closed == 0


def test_model_reference_error_invalidates(harness, monkeypatch):
    def failing_resolve(repo_id, revision="main", *, allow_local=False):
        raise VideoSubmissionError("no such repo", "model_reference_invalid")

    monkeypatch.setattr(vi_module, "resolve_model_dir", failing_resolve)
    metrics = make_module().validate(make_input())
    assert metrics.invalid_submission is True
    assert metrics.diagnostics["failure_mode"] == "model_reference_invalid"


def test_suite_mismatch_is_recoverable_and_touches_no_submission_code(harness):
    harness.package = FakePackage(make_manifest(suite_version="video_inconsistency_v0"), Path("."))
    with pytest.raises(RecoverableException):
        make_module().validate(make_input())
    assert harness.calls["model_dir"] == 0
    assert harness.calls["load"] == 0


def test_max_clips_limits_evaluation(harness):
    metrics = make_module(max_clips=3).validate(make_input())
    assert metrics.clips_evaluated == 3
    assert sorted(harness.detector.calls) == ["clip00", "clip01", "clip02"]
    assert json.loads(metrics.diagnostics["failure_counts"]) == {}


def test_decode_mismatch_is_an_infra_error_and_closes_detector(harness):
    harness.decode_frames = FRAMES - 1
    with pytest.raises(ValueError, match="manifest"):
        make_module().validate(make_input())
    assert harness.detector.closed == 1


def test_unexpected_detector_exception_propagates_and_closes(harness):
    def behaviour(d, cid, attempt):
        raise RuntimeError("disk on fire")

    harness.detector.behaviour = behaviour
    with pytest.raises(RuntimeError):
        make_module().validate(make_input())
    assert harness.detector.closed == 1


def test_detector_receives_only_frames_path_and_fps(harness):
    make_module(max_clips=1).validate(make_input())
    assert harness.detector.detect_kwargs == [{"frames", "video_path", "fps"}]


def test_confidence_below_threshold_is_ignored_by_module_scoring(harness):
    def behaviour(d, cid, attempt):
        answer = d.perfect_answer(cid)
        for issue in answer["issues"]:
            issue["confidence"] = 0.2
        return answer

    harness.detector.behaviour = behaviour
    metrics = make_module().validate(make_input())
    assert metrics.macro_f1 == 0.0
    # Rank-based mAP does not use the confidence threshold: low-confidence but correct
    # answers still earn AP; localization and clip accuracy (threshold-based) do not.
    assert metrics.mean_ap == pytest.approx(1.0)
    assert metrics.localization_score == 0.0
    assert metrics.score == pytest.approx(0.75 + 0.2 * 0.0 + 0.05 * 0.5)
    assert make_module(confidence_threshold=0.1).validate(make_input()).score == pytest.approx(1.0)


def test_local_validation_allows_a_local_submission_dir(harness, monkeypatch):
    from validator.modules.video_inconsistency import local_validate

    seen = {}

    def fake_validate(self, data, **kwargs):
        seen["allow_local"] = self.config.allow_local_model_dir
        return make_module()._invalid_metrics("stub")

    monkeypatch.setattr(vi_module.VideoInconsistencyValidationModule, "validate", fake_validate)
    local_validate.run_local_validation("./my_submission", validation_data_url="/tmp/pkg.zip")
    assert seen["allow_local"] is True
    assert VideoInconsistencyConfig().allow_local_model_dir is False
