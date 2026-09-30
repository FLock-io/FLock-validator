"""Validation module for the video inconsistency detection task.

Pipeline: resolve the validation package (infra) -> resolve the submission repo ->
load the trainer's detector inside the sandbox -> decode each clip on the host and
ask the detector for its findings -> strictly parse the answers -> score them
against the hidden labels.

Error policy: anything that is the *submitter's* fault
raises ``VideoSubmissionError`` and is turned into an invalid (score 0) result;
anything else (broken package, missing sandbox support, disk/network trouble)
propagates so the runner can retry or re-queue instead of zeroing a trainer.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import Field, model_validator

from validator.modules.base import (
    BaseConfig,
    BaseInputData,
    BaseMetrics,
    BaseValidationModule,
)
from validator.modules.video_inconsistency.detector import (
    DEFAULT_ADAPTER_FILENAME,
    load_detector_from_adapter,
    resolve_model_dir,
)
from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.issue_types import SUITE_VERSION
from validator.modules.video_inconsistency.manifest import ClipSpec
from validator.modules.video_inconsistency.package import (
    DEFAULT_PACKAGE_CACHE_DIR,
    ResolvedVideoPackage,
    resolve_validation_package,
)
from validator.modules.video_inconsistency.predictions import (
    PredictedIssue,
    parse_detector_output,
)
from validator.modules.video_inconsistency.scoring import (
    DEFAULT_DIFFICULTY_WEIGHTS,
    DEFAULT_TIOU_THRESHOLDS,
    ScoringSettings,
    score_predictions,
)
from validator.modules.video_inconsistency.video_io import decode_video


WORST_POSSIBLE_LOSS = 1.0
PROGRESS_LOG_EVERY = 10
_URL_FIELDS = ("validation_data_url", "validation_zip_url", "validation_set_url")
# Repository root; its .env holds validator credentials.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _decoy_counts(clips: list[ClipSpec]) -> dict[str, int]:
    """Decoys (legitimate, unlabelled events) per type in the evaluated clips.

    Telemetry only: decoys are hard negatives and never enter the score.
    """
    counts: Counter[str] = Counter()
    for clip in clips:
        counts.update(decoy.type for decoy in getattr(clip, "decoys", ()))
    return dict(counts)


class VideoInconsistencyConfig(BaseConfig):
    suite_version: str = SUITE_VERSION
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    package_cache_dir: str = DEFAULT_PACKAGE_CACHE_DIR
    max_clips: int | None = Field(default=None, ge=1)

    detector_load_timeout_seconds: float = Field(default=600.0, gt=0)
    detect_timeout_seconds: float = Field(default=60.0, gt=0)
    # Authoritative model-size ceiling, enforced at runtime by the sandbox (CUDA
    # allocator fraction + host RSS monitor), however the weights are represented.
    detector_memory_limit_gb: int = Field(default=18, ge=1)
    # Floor for the worker's RLIMIT_CPU. The effective budget is scaled up to cover
    # the run's worst-case wall time on every core (see _cpu_time_budget), so it is
    # only a backstop and never kills a detector that is within its wall-time limits.
    detector_cpu_time_seconds: int = Field(default=7200, ge=1)
    # Local validation only: accept a local directory as the submission. Production
    # references come from trainers and must never resolve to host paths.
    allow_local_model_dir: bool = False
    # Extra attempts (after the first) for a clip whose detect() raised or
    # returned malformed output while the sandbox is still healthy.
    detect_retries: int = Field(default=2, ge=0)
    # The submission is invalid once more than this share of planned clips failed.
    max_failed_clip_fraction: float = Field(default=0.25, ge=0.0, le=1.0)

    # Scoring settings (see scoring.py); exposed so operators can tune them.
    # mAP is averaged over these tIoU thresholds; ``tiou_threshold`` only drives the
    # F1 / localization reporting terms.
    tiou_thresholds: list[float] = Field(
        default_factory=lambda: list(DEFAULT_TIOU_THRESHOLDS), min_length=1
    )
    tiou_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    bbox_iou_threshold: float = Field(default=0.3, gt=0.0, le=1.0)
    min_event_seconds: float = Field(default=0.4, ge=0.0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_predictions_per_clip: int = Field(default=25, ge=1)
    difficulty_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_DIFFICULTY_WEIGHTS)
    )
    weight_map: float = Field(default=0.75, ge=0.0, le=1.0)
    weight_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    weight_localization: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_clip_accuracy: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _scoring_settings_are_consistent(self) -> "VideoInconsistencyConfig":
        # Fail at config-load time (weights must sum to 1, all difficulties present)
        # rather than after a submission has been loaded into a sandbox.
        self.scoring_settings()
        return self

    def scoring_settings(self) -> ScoringSettings:
        return ScoringSettings(
            tiou_thresholds=tuple(self.tiou_thresholds),
            tiou_threshold=self.tiou_threshold,
            bbox_iou_threshold=self.bbox_iou_threshold,
            min_event_seconds=self.min_event_seconds,
            confidence_threshold=self.confidence_threshold,
            max_predictions_per_clip=self.max_predictions_per_clip,
            difficulty_weights=tuple(self.difficulty_weights.items()),
            weight_map=self.weight_map,
            weight_f1=self.weight_f1,
            weight_localization=self.weight_localization,
            weight_clip_accuracy=self.weight_clip_accuracy,
        )


class VideoInconsistencyInputData(BaseInputData):
    hg_repo_id: str
    revision: str = "main"
    validation_data_url: str | None = None
    validation_zip_url: str | None = None
    validation_set_url: str | None = None
    adapter_filename: str = DEFAULT_ADAPTER_FILENAME


class VideoInconsistencyMetrics(BaseMetrics):
    score: float
    loss: float
    macro_f1: float
    micro_f1: float
    precision: float
    recall: float
    localization_score: float
    clip_accuracy: float
    # Rank-based headline term: mean average precision over issue types and tIoU
    # thresholds, with its per-type and per-threshold breakdowns.
    mean_ap: float
    per_type_ap: dict[str, float] = Field(default_factory=dict)
    ap_by_tiou: dict[str, float] = Field(default_factory=dict)
    per_type_f1: dict[str, float] = Field(default_factory=dict)
    # Clips that were run through the detector (failed ones included, scored as
    # empty answers); ``clips_failed`` is the subset whose detection failed.
    clips_evaluated: int
    clips_failed: int
    invalid_submission: bool = False
    parameter_count: int | None = None
    diagnostics: dict[str, str] = Field(default_factory=dict)


class VideoInconsistencyValidationModule(BaseValidationModule):
    config_schema = VideoInconsistencyConfig
    metrics_schema = VideoInconsistencyMetrics
    input_data_schema = VideoInconsistencyInputData
    task_type = "video_inconsistency"

    def __init__(self, config: VideoInconsistencyConfig, **kwargs):
        self.config = config

    # ------------------------------------------------------------------ validate

    def validate(
        self, data: VideoInconsistencyInputData, **kwargs
    ) -> VideoInconsistencyMetrics:
        parameter_count: int | None = None
        try:
            # Resolve the package first so operator-side problems surface before
            # any submission code is touched.
            package = self._resolve_package(data)
            if package.manifest.suite_version != self.config.suite_version:
                # Package/config mismatch is an operator problem, not the
                # submitter's: let the assignment be re-queued rather than zeroed.
                from validator.exceptions import RecoverableException

                raise RecoverableException(
                    f"Manifest suite_version {package.manifest.suite_version!r} does not "
                    f"match config suite_version {self.config.suite_version!r}"
                )

            model_dir = resolve_model_dir(
                data.hg_repo_id,
                data.revision,
                allow_local=self.config.allow_local_model_dir,
            )
            detector = load_detector_from_adapter(
                model_dir,
                data.adapter_filename,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                load_timeout_seconds=self.config.detector_load_timeout_seconds,
                detect_timeout_seconds=self.config.detect_timeout_seconds,
                memory_limit_bytes=self.config.detector_memory_limit_gb * 1024**3,
                cpu_time_seconds=self._cpu_time_budget(len(self._planned_clips(package))),
                # Hidden labels and validator credentials. Landlock already keeps
                # them out of reach on Linux; this also covers macOS hosts.
                deny_read_paths=[package.root, _REPO_ROOT / ".env"],
            )
            try:
                parameter_count = getattr(detector, "parameter_count", None)
                return self._evaluate(package, detector, parameter_count)
            finally:
                detector.close()
        except VideoSubmissionError as exc:
            # The submission is invalid/unrunnable: score it 0 and keep the
            # validator alive. Infra errors are deliberately not caught here.
            logger.error(f"Invalid video inconsistency submission [{exc.failure_mode}]: {exc}")
            return self._invalid_metrics(
                str(exc), parameter_count=parameter_count, failure_mode=exc.failure_mode
            )

    def _resolve_package(self, data: VideoInconsistencyInputData) -> ResolvedVideoPackage:
        url = next((getattr(data, name) for name in _URL_FIELDS if getattr(data, name)), None)
        if not url:
            raise ValueError(
                "Video inconsistency validation requires validation_data_url, "
                "validation_zip_url or validation_set_url"
            )
        return resolve_validation_package(str(url), self.config.package_cache_dir)

    # ------------------------------------------------------------------ evaluation

    def _evaluate(
        self,
        package: ResolvedVideoPackage,
        detector: Any,
        parameter_count: int | None,
    ) -> VideoInconsistencyMetrics:
        clips = self._planned_clips(package)
        planned = len(clips)
        failure_modes: Counter[str] = Counter()
        evaluated: list[ClipSpec] = []
        predictions: list[list[PredictedIssue]] = []
        detect_seconds = 0.0
        started = time.perf_counter()

        for index, clip in enumerate(clips, start=1):
            frames = self._decode_clip(package, clip)
            detect_started = time.perf_counter()
            clip_predictions, failure_mode = self._detect_with_retries(
                detector, package, clip, frames
            )
            detect_seconds += time.perf_counter() - detect_started
            evaluated.append(clip)
            predictions.append(clip_predictions)
            if failure_mode is not None:
                failure_modes[failure_mode] += 1
                self._raise_if_too_many_failures(failure_modes, planned)
            if index % PROGRESS_LOG_EVERY == 0 or index == planned:
                logger.info(
                    f"Video inconsistency progress: {index}/{planned} clips, "
                    f"{sum(failure_modes.values())} failed"
                )

        result = score_predictions(evaluated, predictions, self.config.scoring_settings())
        diagnostics = {
            **package.diagnostics,
            "clips_planned": str(planned),
            "failure_counts": json.dumps(dict(failure_modes), sort_keys=True),
            "per_type_counts": json.dumps(result.per_type_counts, sort_keys=True),
            "decoy_counts": json.dumps(_decoy_counts(evaluated), sort_keys=True),
            "detect_seconds": f"{detect_seconds:.2f}",
            "total_seconds": f"{time.perf_counter() - started:.2f}",
            "parameter_count": str(parameter_count),
        }
        return VideoInconsistencyMetrics(
            score=result.score,
            loss=result.loss,
            macro_f1=result.macro_f1,
            micro_f1=result.micro_f1,
            precision=result.precision,
            recall=result.recall,
            localization_score=result.localization_score,
            clip_accuracy=result.clip_accuracy,
            mean_ap=result.mean_ap,
            per_type_ap=result.per_type_ap,
            ap_by_tiou=result.ap_by_tiou,
            per_type_f1=result.per_type_f1,
            clips_evaluated=len(evaluated),
            clips_failed=sum(failure_modes.values()),
            invalid_submission=False,
            parameter_count=parameter_count,
            diagnostics=diagnostics,
        )

    def _planned_clips(self, package: ResolvedVideoPackage) -> list[ClipSpec]:
        clips = list(package.manifest.clips)
        if self.config.max_clips is not None:
            clips = clips[: self.config.max_clips]
        return clips

    def _cpu_time_budget(self, planned_clips: int) -> int:
        """RLIMIT_CPU for the worker: a backstop, never tighter than the wall limits.

        RLIMIT_CPU sums CPU time over every thread, so a multi-threaded detector on
        a many-core host burns it many times faster than wall time. Scale it to the
        run's worst-case wall time (load plus every attempt on every clip) on all
        cores, with the configured value as a floor.
        """
        attempts = self.config.detect_retries + 1
        worst_wall_seconds = (
            self.config.detector_load_timeout_seconds
            + planned_clips * attempts * self.config.detect_timeout_seconds
        )
        scaled = math.ceil(worst_wall_seconds * (os.cpu_count() or 1))
        return max(self.config.detector_cpu_time_seconds, scaled)

    def _decode_clip(self, package: ResolvedVideoPackage, clip: ClipSpec) -> Any:
        """Decode on the host. A mismatch with the manifest is a package (infra) bug."""
        path = package.clip_video_path(clip)
        # Bound memory: one frame beyond the manifest is enough to expose a mismatch.
        frames = decode_video(path, max_frames=clip.num_frames + 1)
        if frames.shape[0] != clip.num_frames:
            raise ValueError(
                f"clip {clip.clip_id}: decoded {frames.shape[0]} frames, "
                f"manifest says {clip.num_frames}"
            )
        if frames.shape[1] != clip.height or frames.shape[2] != clip.width:
            raise ValueError(
                f"clip {clip.clip_id}: decoded {frames.shape[2]}x{frames.shape[1]}, "
                f"manifest says {clip.width}x{clip.height}"
            )
        return frames

    def _detect_with_retries(
        self,
        detector: Any,
        package: ResolvedVideoPackage,
        clip: ClipSpec,
        frames: Any,
    ) -> tuple[list[PredictedIssue], str | None]:
        """Return ``(predictions, failure_mode)``.

        Non-fatal errors are retried; when they persist the clip is recorded as a
        failure with empty predictions. Fatal errors propagate to abort the run.
        """
        video_path = package.clip_video_path(clip)
        attempts = self.config.detect_retries + 1
        last_error: VideoSubmissionError | None = None
        for attempt in range(1, attempts + 1):
            try:
                raw = detector.detect(frames=frames, video_path=video_path, fps=clip.fps)
                return parse_detector_output(raw, clip.duration), None
            except VideoSubmissionError as exc:
                if exc.fatal:
                    raise
                last_error = exc
                logger.warning(
                    f"Clip {clip.clip_id} attempt {attempt}/{attempts} failed "
                    f"[{exc.failure_mode}]: {exc}"
                )
        assert last_error is not None
        return [], last_error.failure_mode

    def _raise_if_too_many_failures(self, failure_modes: Counter[str], planned: int) -> None:
        failed = sum(failure_modes.values())
        if failed <= self.config.max_failed_clip_fraction * planned:
            return
        dominant, _ = failure_modes.most_common(1)[0]
        raise VideoSubmissionError(
            f"{failed} of {planned} clips failed, above the allowed fraction "
            f"{self.config.max_failed_clip_fraction}; most common failure: {dominant}",
            dominant,
            fatal=True,
        )

    # ------------------------------------------------------------------ helpers

    def _invalid_metrics(
        self,
        reason: str,
        parameter_count: int | None = None,
        failure_mode: str = "submission_error",
    ) -> VideoInconsistencyMetrics:
        return VideoInconsistencyMetrics(
            score=0.0,
            loss=WORST_POSSIBLE_LOSS,
            macro_f1=0.0,
            micro_f1=0.0,
            precision=0.0,
            recall=0.0,
            localization_score=0.0,
            clip_accuracy=0.0,
            mean_ap=0.0,
            per_type_ap={},
            ap_by_tiou={},
            per_type_f1={},
            clips_evaluated=0,
            clips_failed=0,
            invalid_submission=True,
            parameter_count=parameter_count,
            diagnostics={"reason": reason, "failure_mode": failure_mode},
        )

    def cleanup(self):
        pass


MODULE = VideoInconsistencyValidationModule
