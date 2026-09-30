"""Host side of the sandboxed video detector.

``load_detector_from_adapter`` starts ``detector_worker.py`` inside a
``SandboxProcess`` and returns an ``IsolatedDetector`` proxy. The trainer's code
never runs in this process; only bounded JSON crosses the boundary, and video
frames are handed over as files in a host-owned, worker-read-only directory.
"""

from __future__ import annotations

import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from huggingface_hub import errors as hf_errors
from huggingface_hub import snapshot_download
from loguru import logger

from validator.exceptions import RecoverableException
from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.issue_types import ISSUE_TYPE_NAMES
from validator.sandbox import (
    SandboxError,
    SandboxProcess,
    SandboxUnavailableError,
    protect_parent_secrets,
)

DEFAULT_ADAPTER_FILENAME = "flock_video_adapter.py"
DEFAULT_DETECTOR_LOAD_TIMEOUT_SECONDS = 10 * 60
DEFAULT_DETECTOR_DETECT_TIMEOUT_SECONDS = 120
# Authoritative model-size ceiling for the detector sandbox (GPU VRAM + system RAM).
DEFAULT_DETECTOR_MEMORY_LIMIT_BYTES = 18 * 1024**3
DEFAULT_DETECTOR_CPU_TIME_SECONDS = 60 * 60

# Escape hatch for local runs on macOS hosts that cannot nest sandbox-exec.
UNSAFE_LOCAL_ENV_VAR = "VIDEO_INCONSISTENCY_ALLOW_UNSAFE_LOCAL_ADAPTER"

_FRAMES_FILENAME = "frames.npy"
# Neutral name: the package's clip id / filename must never reach the trainer.
_VIDEO_FILENAME = "input.mp4"
_MAX_ERROR_CHARS = 500

# Failure modes a worker reply may carry. Anything else came from a confused or
# hostile worker and is replaced by a fixed fallback instead of being surfaced.
_NON_FATAL_WORKER_MODES = frozenset(
    {"detector_execution_failed", "detector_output_invalid"}
)
_FATAL_WORKER_MODES = frozenset(
    {
        "adapter_contract",
        "adapter_import_failed",
        "model_load_failed",
        "detector_protocol_error",
    }
)

_LOAD_SANDBOX_MODES = {
    "model_load_timeout": "model_load_timeout",
    "memory_exceeded": "detector_memory_exceeded",
    # A worker dying while loading is a failed load, not a serving crash.
    "crashed": "model_load_failed",
    "protocol_error": "detector_protocol_error",
}
_DETECT_SANDBOX_MODES = {
    "detector_timeout": "detector_timeout",
    "memory_exceeded": "detector_memory_exceeded",
    "crashed": "detector_crashed",
    "protocol_error": "detector_protocol_error",
}


def resolve_model_dir(
    repo_id_or_path: str, revision: str = "main", *, allow_local: bool = False
) -> Path:
    """Download the submitted Hugging Face repo (or use a local dir when allowed).

    ``allow_local`` is only for local validation. In production the reference
    comes from the trainer, and treating it as a host path would let a submission
    point the sandbox (which is granted read access to the model directory) at
    arbitrary validator files such as the extracted validation package.
    """
    if allow_local:
        candidate = Path(repo_id_or_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

    token = os.getenv("HF_TOKEN")
    try:
        path = snapshot_download(
            repo_id=repo_id_or_path, revision=revision, token=token
        )
    except Exception as exc:
        if not _is_deterministic_hub_error(exc):
            raise
        raise VideoSubmissionError(
            f"Invalid Hugging Face model reference {repo_id_or_path!r} "
            f"at revision {revision!r}: {exc}",
            failure_mode="model_reference_invalid",
        ) from exc
    return Path(path).resolve()


def _is_deterministic_hub_error(exc: Exception) -> bool:
    """Return whether a Hub failure is caused by the submitted reference.

    Connection failures, server failures, request timeouts, and rate limits stay
    recoverable. Malformed IDs and stable 4xx responses are invalid submissions.
    """
    local_miss_type = getattr(hf_errors, "LocalEntryNotFoundError", None)
    if isinstance(local_miss_type, type) and isinstance(exc, local_miss_type):
        return False

    deterministic_types = tuple(
        error_type
        for error_type in (
            getattr(hf_errors, "HFValidationError", None),
            getattr(hf_errors, "RepositoryNotFoundError", None),
            getattr(hf_errors, "RevisionNotFoundError", None),
            getattr(hf_errors, "EntryNotFoundError", None),
            getattr(hf_errors, "GatedRepoError", None),
            getattr(hf_errors, "DisabledRepoError", None),
            getattr(hf_errors, "BadRequestError", None),
        )
        if isinstance(error_type, type)
    )
    if isinstance(exc, deterministic_types):
        return True

    http_error_type = getattr(hf_errors, "HfHubHTTPError", None)
    if isinstance(http_error_type, type) and isinstance(exc, http_error_type):
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        return (
            isinstance(status_code, int)
            and 400 <= status_code < 500
            and status_code not in {408, 429}
        )
    return False


def _check_adapter_path(model_root: Path, adapter_filename: str) -> Path:
    adapter_path = model_root / adapter_filename
    # Lexical check: guard against adapter_filename escaping via '..' or an
    # absolute path. We normalise without resolving symlinks here because
    # HuggingFace snapshots store repo files as symlinks into a sibling blobs/
    # directory — resolving would wrongly flag every real HF download.
    normalized = os.path.normpath(str(adapter_path))
    root_str = str(model_root)
    if normalized != root_str and not normalized.startswith(root_str + os.sep):
        raise VideoSubmissionError(
            "adapter_filename must resolve inside the model directory",
            failure_mode="adapter_contract",
        )
    if not adapter_path.is_file():
        raise VideoSubmissionError(
            f"Missing video detector adapter: {adapter_path}",
            failure_mode="adapter_missing",
        )
    # Symlink check: if the adapter is a symlink (e.g. HF hub cache links blobs/
    # next to snapshots/), ensure the real target stays within the model cache
    # root — not an arbitrary filesystem path a trainer could embed via git symlink.
    if adapter_path.is_symlink():
        resolved_target = adapter_path.resolve()
        # HF cache layout: model_root = .../models--org--repo/snapshots/<hash>/
        # Blobs live at   : .../models--org--repo/blobs/<sha256>
        # Allow the target to be anywhere within the repo-level cache directory.
        allowed_root = model_root.parent.parent
        if not (
            str(resolved_target) == str(model_root)
            or str(resolved_target).startswith(str(model_root) + os.sep)
            or str(resolved_target).startswith(str(allowed_root) + os.sep)
        ):
            raise VideoSubmissionError(
                f"Adapter {adapter_filename!r} is a symlink to a path outside "
                "the model directory",
                failure_mode="adapter_symlink_escape",
            )
    return adapter_path


def load_detector_from_adapter(
    model_dir: Path,
    adapter_filename: str,
    *,
    device: str,
    torch_dtype: str,
    load_timeout_seconds: float = DEFAULT_DETECTOR_LOAD_TIMEOUT_SECONDS,
    detect_timeout_seconds: float = DEFAULT_DETECTOR_DETECT_TIMEOUT_SECONDS,
    memory_limit_bytes: int = DEFAULT_DETECTOR_MEMORY_LIMIT_BYTES,
    cpu_time_seconds: int = DEFAULT_DETECTOR_CPU_TIME_SECONDS,
    deny_read_paths: Sequence[Path] = (),
) -> "IsolatedDetector":
    model_root = model_dir.resolve()
    _check_adapter_path(model_root, adapter_filename)
    return IsolatedDetector.start(
        model_dir=model_root,
        adapter_filename=adapter_filename,
        device=device,
        torch_dtype=torch_dtype,
        load_timeout_seconds=load_timeout_seconds,
        detect_timeout_seconds=detect_timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        cpu_time_seconds=cpu_time_seconds,
        deny_read_paths=deny_read_paths,
    )


class IsolatedDetector:
    """Safe proxy for a trainer's detector running in a sandbox process."""

    def __init__(
        self,
        process: SandboxProcess,
        input_dir: Path,
        detect_timeout_seconds: float,
    ) -> None:
        self._process = process
        self._input_dir = input_dir
        self._detect_timeout_seconds = detect_timeout_seconds
        # Telemetry only (int, or None when unknown). Model size is enforced by the
        # sandbox memory monitor, not this count.
        self.parameter_count: int | None = None
        self._closed = False

    @classmethod
    def start(
        cls,
        *,
        model_dir: Path,
        adapter_filename: str,
        device: str,
        torch_dtype: str,
        load_timeout_seconds: float,
        detect_timeout_seconds: float,
        memory_limit_bytes: int,
        cpu_time_seconds: int,
        deny_read_paths: Sequence[Path] = (),
    ) -> "IsolatedDetector":
        if (
            min(
                load_timeout_seconds,
                detect_timeout_seconds,
                memory_limit_bytes,
                cpu_time_seconds,
            )
            <= 0
        ):
            raise ValueError("Detector sandbox limits must all be positive")

        try:
            protect_parent_secrets()
        except SandboxUnavailableError as exc:
            raise RecoverableException(str(exc)) from exc

        # Clips are staged in a host-owned directory that is NOT the worker's
        # writable TMPDIR. The worker gets it read-only (Landlock), so it can
        # neither alter frames it is handed nor plant files the host might read.
        input_dir = Path(tempfile.mkdtemp(prefix="video-detector-input-"))  # 0700
        try:
            process = SandboxProcess(
                _worker_command(
                    model_dir=model_dir,
                    input_dir=input_dir,
                    adapter_filename=adapter_filename,
                    device=device,
                    torch_dtype=torch_dtype,
                    memory_limit_bytes=memory_limit_bytes,
                    cpu_time_seconds=cpu_time_seconds,
                ),
                memory_limit_bytes=memory_limit_bytes,
                temp_prefix="video-detector-worker-",
                unsafe_local_env_var=UNSAFE_LOCAL_ENV_VAR,
                deny_read_paths=deny_read_paths,
            )
        except SandboxUnavailableError as exc:
            shutil.rmtree(input_dir, ignore_errors=True)
            raise RecoverableException(str(exc)) from exc
        except Exception:
            shutil.rmtree(input_dir, ignore_errors=True)
            raise

        detector = cls(process, input_dir, detect_timeout_seconds)
        try:
            try:
                response = process.receive(load_timeout_seconds, "model_load_timeout")
            except SandboxError as exc:
                raise _translate_sandbox_error(exc, _LOAD_SANDBOX_MODES) from exc
            detector._raise_for_load_reply(response)
            count = response.get("parameter_count")
            valid_count = (
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
            )
            detector.parameter_count = count if valid_count else None
            return detector
        except Exception:
            detector.close()
            raise

    def detect(self, *, frames: np.ndarray, video_path: Path, fps: float) -> Any:
        if self._closed:
            raise VideoSubmissionError(
                "Detector sandbox is no longer running", failure_mode="detector_crashed"
            )
        num_frames, height, width = _validate_frames(frames)
        if not fps > 0:
            raise ValueError("fps must be positive")

        clip_dir = stage_clip(self._input_dir, frames, Path(video_path))
        try:
            message = {
                "op": "detect",
                "frames_path": str(clip_dir / _FRAMES_FILENAME),
                "video_path": str(clip_dir / _VIDEO_FILENAME),
                "fps": float(fps),
                "num_frames": num_frames,
                "width": width,
                "height": height,
                "duration": num_frames / float(fps),
                "issue_types": list(ISSUE_TYPE_NAMES),
            }
            try:
                response = self._process.request(
                    message, self._detect_timeout_seconds, "detector_timeout"
                )
            except SandboxError as exc:
                self.close()
                raise _translate_sandbox_error(exc, _DETECT_SANDBOX_MODES) from exc
            return self._result_from_reply(response)
        finally:
            shutil.rmtree(clip_dir, ignore_errors=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._process.close()
        finally:
            shutil.rmtree(self._input_dir, ignore_errors=True)

    def __enter__(self) -> "IsolatedDetector":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finaliser
            pass

    def _raise_for_load_reply(self, response: dict[str, Any]) -> None:
        if response.get("ok") is True:
            return
        mode = response.get("failure_mode")
        message = _clip_text(response.get("error"), "Detector sandbox rejected the load")
        if mode == "sandbox_unavailable":
            # Only honoured before any trainer code has answered a request: the
            # worker reports it from its trusted hardening step, which runs
            # before trainer code is imported.
            raise RecoverableException(f"Detector sandbox unavailable: {message}")
        if mode not in _FATAL_WORKER_MODES:
            mode = "model_load_failed"
        raise VideoSubmissionError(message, failure_mode=str(mode))

    def _result_from_reply(self, response: dict[str, Any]) -> Any:
        if response.get("ok") is True and "result" in response:
            return response["result"]
        mode = response.get("failure_mode")
        message = _clip_text(response.get("error"), "Detector sandbox rejected the request")
        if mode in _NON_FATAL_WORKER_MODES:
            raise VideoSubmissionError(message, failure_mode=str(mode), fatal=False)
        # A reply that is neither a result nor a known per-clip error means the
        # worker is out of sync (or hostile). `sandbox_unavailable` is deliberately
        # NOT honoured here: trainer code is running, so it could forge it to force
        # endless retries instead of a score.
        self.close()
        if mode not in _FATAL_WORKER_MODES:
            mode = "detector_protocol_error"
        raise VideoSubmissionError(message, failure_mode=str(mode))


def _worker_command(
    *,
    model_dir: Path,
    input_dir: Path,
    adapter_filename: str,
    device: str,
    torch_dtype: str,
    memory_limit_bytes: int,
    cpu_time_seconds: int,
) -> list[str]:
    worker = Path(__file__).with_name("detector_worker.py")
    return [
        sys.executable,
        "-u",
        str(worker),
        "--model-dir",
        str(model_dir),
        "--input-dir",
        str(input_dir),
        "--adapter-filename",
        adapter_filename,
        "--device",
        device,
        "--torch-dtype",
        torch_dtype,
        "--memory-limit-bytes",
        str(memory_limit_bytes),
        "--cpu-time-seconds",
        str(cpu_time_seconds),
    ]


def _translate_sandbox_error(
    exc: SandboxError, mapping: dict[str, str]
) -> VideoSubmissionError:
    mode = mapping.get(exc.failure_mode, "detector_protocol_error")
    logger.warning(f"Detector sandbox failure ({exc.failure_mode} -> {mode}): {exc}")
    return VideoSubmissionError(str(exc), failure_mode=mode, fatal=True)


def _clip_text(value: Any, default: str) -> str:
    text = str(value) if value else default
    return text[:_MAX_ERROR_CHARS]


def _validate_frames(frames: np.ndarray) -> tuple[int, int, int]:
    if not isinstance(frames, np.ndarray):
        raise ValueError("frames must be a numpy array")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"frames must be uint8 (T, H, W, 3), got {frames.dtype} {frames.shape}"
        )
    if frames.shape[0] < 1:
        raise ValueError("frames must contain at least one frame")
    if not frames.flags["C_CONTIGUOUS"]:
        raise ValueError("frames must be C-contiguous")
    return int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])


def _write_new_file(path: Path, writer: Callable[[Any], None]) -> None:
    """Create ``path`` exclusively and write it via ``writer(binary_file)``.

    O_EXCL fails if anything (including a planted symlink or dangling symlink)
    already exists at ``path``, and O_NOFOLLOW refuses to follow a symlink at the
    final component, so a host write can never be redirected to another file.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        writer(handle)


def stage_clip(input_dir: Path, frames: np.ndarray, video_path: Path) -> Path:
    """Write one clip into a fresh unpredictable subdirectory of ``input_dir``.

    Returns the subdirectory, containing ``frames.npy`` and ``input.mp4`` only.
    Names are neutral so nothing about the clip's identity reaches the trainer.
    The caller deletes the subdirectory when the request is done.
    """
    _validate_frames(frames)
    clip_dir = input_dir / secrets.token_hex(16)
    # mkdir fails with FileExistsError on anything already at that path, including
    # a symlink, so the clip directory is guaranteed to be newly created by us.
    os.mkdir(clip_dir, 0o700)
    try:
        _write_new_file(
            clip_dir / _FRAMES_FILENAME,
            lambda handle: np.lib.format.write_array(handle, frames, allow_pickle=False),
        )

        def copy_video(handle: Any) -> None:
            with open(video_path, "rb") as source:
                shutil.copyfileobj(source, handle)

        _write_new_file(clip_dir / _VIDEO_FILENAME, copy_video)
    except BaseException:
        shutil.rmtree(clip_dir, ignore_errors=True)
        raise
    return clip_dir
