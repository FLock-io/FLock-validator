"""Sandbox worker: loads a trainer's detector and serves ``detect`` requests.

This process runs UNTRUSTED trainer code. It is started by
``detector.IsolatedDetector`` inside ``validator.sandbox.SandboxProcess`` and
speaks length-prefixed JSON on stdin/stdout.

Trust invariant: every hardening step (rlimits, Landlock, seccomp, CUDA cap) runs
before any trainer code is imported. A hardening failure is therefore reported as
``sandbox_unavailable`` (an infrastructure problem: the host cannot sandbox), and
that failure mode can only originate from this trusted startup path. Once trainer
code runs, every exception it raises is re-wrapped with a fixed failure mode, so a
submission cannot fake ``sandbox_unavailable`` to dodge scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


# Running this file directly puts video_inconsistency/ on sys.path, not the
# repository root. Add the root explicitly before importing trusted validator
# modules.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Everything the worker needs from the repository is imported HERE, before
# hardening: once Landlock is installed the checkout is unreadable.
from validator.sandbox.hardening import (  # noqa: E402
    apply_cuda_memory_limit,
    apply_resource_limits,
    install_landlock,
    install_seccomp,
)
from validator.sandbox.protocol import (  # noqa: E402
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MessageTooLargeError,
    read_message_blocking,
    to_jsonable,
    write_message,
)

# The repo is read-only from here on and bytecode caches next to submitted code
# must not be written (or attempted).
sys.dont_write_bytecode = True

_MAX_ERROR_CHARS = 500
_LOAD_ATTEMPTS = 2


class _WorkerFailure(Exception):
    """A failure the worker itself classifies; carries a trusted failure mode."""

    def __init__(self, message: str, failure_mode: str) -> None:
        super().__init__(message)
        self.failure_mode = failure_mode


def main() -> None:
    args = _parse_args()
    protocol_in = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    protocol_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    os.close(devnull)

    try:
        _harden(args)
    except Exception as exc:  # noqa: BLE001 - report hardening failures
        # No trainer code has run yet, so this classification is trustworthy.
        _write_error(protocol_out, exc, "sandbox_unavailable")
        return

    try:
        detector = _load_detector(args)
        parameter_count = _count_parameters_safe(detector)
        write_message(
            protocol_out,
            {"ok": True, "parameter_count": parameter_count},
            max_bytes=MAX_RESPONSE_BYTES,
        )
    except _WorkerFailure as exc:
        _write_error(protocol_out, exc, "model_load_failed")
        return
    except Exception as exc:  # noqa: BLE001 - defensive: never crash silently
        _write_error(protocol_out, exc, "model_load_failed")
        return

    while True:
        try:
            message = read_message_blocking(protocol_in, max_bytes=MAX_REQUEST_BYTES)
        except EOFError:
            return
        except Exception as exc:  # noqa: BLE001 - malformed request from the host
            _write_error(protocol_out, exc, "detector_protocol_error")
            return
        operation = message.get("op")
        if operation == "close":
            return
        if operation != "detect":
            _write_error(
                protocol_out,
                _WorkerFailure(
                    f"Unsupported detector operation {operation!r}",
                    "detector_protocol_error",
                ),
                "detector_protocol_error",
            )
            continue
        _serve_detect(protocol_out, detector, message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--adapter-filename", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--torch-dtype", required=True)
    parser.add_argument("--memory-limit-bytes", type=int, required=True)
    parser.add_argument("--cpu-time-seconds", type=int, required=True)
    return parser.parse_args()


def _harden(args: argparse.Namespace) -> None:
    apply_resource_limits(args.memory_limit_bytes, args.cpu_time_seconds, args.device)
    install_landlock(
        Path(args.model_dir),
        # Host-staged clips: readable, never writable, so a worker cannot alter
        # the frames it is later handed or plant files the host will read.
        extra_read_only_dirs=[Path(args.input_dir)],
        temp_dir=Path(os.environ["TMPDIR"]),
    )
    install_seccomp()
    apply_cuda_memory_limit(args.device, args.memory_limit_bytes)


def _load_python_module(path: Path) -> ModuleType:
    module_digest = hashlib.sha256(os.fsencode(path.absolute())).hexdigest()[:16]
    module_name = f"_flock_video_adapter_{module_digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load adapter module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    module_dir = str(path.parent.absolute())
    inserted_module_dir = module_dir not in sys.path
    if inserted_module_dir:
        sys.path.insert(0, module_dir)
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(spec.name) is module:
            del sys.modules[spec.name]
        raise
    finally:
        if inserted_module_dir:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass
    return module


def _describe(exc: BaseException) -> str:
    try:
        text = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001 - a hostile __str__ must not kill the worker
        text = type(exc).__name__
    return text[:_MAX_ERROR_CHARS]


def _load_detector(args: argparse.Namespace) -> Any:
    model_root = Path(args.model_dir).resolve()
    adapter_path = model_root / args.adapter_filename
    try:
        module = _load_python_module(adapter_path)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - untrusted trainer code
        raise _WorkerFailure(
            f"Video adapter failed to import: {_describe(exc)}",
            "adapter_import_failed",
        ) from None
    loader = getattr(module, "load_detector", None)
    if not callable(loader):
        raise _WorkerFailure(
            f"{adapter_path.name} must define "
            "load_detector(model_dir, device, dtype) -> detector",
            "adapter_contract",
        )

    detector: Any = None
    last_error = ""
    for attempt in range(1, _LOAD_ATTEMPTS + 1):
        try:
            detector = loader(
                model_dir=str(model_root),
                device=args.device,
                dtype=args.torch_dtype,
            )
            break
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - untrusted trainer code
            last_error = _describe(exc)
    else:
        raise _WorkerFailure(
            f"load_detector failed after {_LOAD_ATTEMPTS} attempts: {last_error}",
            "model_load_failed",
        )
    if not callable(getattr(detector, "detect", None)):
        raise _WorkerFailure(
            "load_detector must return an object with a callable detect(video)",
            "adapter_contract",
        )
    return detector


def _count_parameters_safe(detector: Any) -> int | None:
    """Best-effort parameter count for telemetry only.

    Model size is enforced at runtime by the host memory monitor and the CUDA
    allocator cap, not by this number, so it never raises and never gates. It
    walks the detector's attributes one and two levels deep for torch modules and
    sums their unique parameters; ``None`` means unknown. torch is only consulted
    if the adapter already imported it, so a numpy-only detector never pays the
    import cost.
    """
    try:
        torch = sys.modules.get("torch")
        if torch is None:
            return None
        module_type = torch.nn.Module
        found: list[Any] = []
        seen_objects: set[int] = set()

        def children(obj: Any) -> list[Any]:
            if isinstance(obj, module_type):
                return []
            values: list[Any] = []
            attrs = getattr(obj, "__dict__", None)
            if isinstance(attrs, dict):
                values.extend(attrs.values())
            if isinstance(obj, dict):
                values.extend(obj.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                values.extend(obj)
            return values

        level = [detector]
        for _depth in range(3):  # the detector itself, then two levels of attributes
            following: list[Any] = []
            for obj in level:
                if id(obj) in seen_objects:
                    continue
                seen_objects.add(id(obj))
                if isinstance(obj, module_type):
                    found.append(obj)
                else:
                    following.extend(children(obj))
            level = following
        if not found:
            return None
        seen_params: set[int] = set()
        total = 0
        for module in found:
            for parameter in module.parameters():
                if id(parameter) not in seen_params:
                    seen_params.add(id(parameter))
                    total += int(parameter.numel())
        return total
    except Exception:  # noqa: BLE001 - telemetry must never break serving
        return None


def _serve_detect(protocol_out: Any, detector: Any, message: dict[str, Any]) -> None:
    try:
        video = _build_video(message)
    except Exception as exc:  # noqa: BLE001 - host-side staging problem
        _write_error(
            protocol_out,
            _WorkerFailure(
                f"Could not open the staged clip: {_describe(exc)}",
                "detector_protocol_error",
            ),
            "detector_protocol_error",
        )
        return

    try:
        result = detector.detect(video)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - untrusted trainer code
        _write_error(
            protocol_out,
            _WorkerFailure(_describe(exc), "detector_execution_failed"),
            "detector_execution_failed",
        )
        return

    try:
        payload = to_jsonable(result)
    except (Exception, SystemExit) as exc:  # noqa: BLE001 - untrusted result object
        _write_error(
            protocol_out,
            _WorkerFailure(
                f"detect() result is not JSON-serialisable: {_describe(exc)}",
                "detector_output_invalid",
            ),
            "detector_output_invalid",
        )
        return

    try:
        write_message(
            protocol_out,
            {"ok": True, "result": payload},
            max_bytes=MAX_RESPONSE_BYTES,
        )
    except MessageTooLargeError:
        _write_error(
            protocol_out,
            _WorkerFailure(
                "detect() result exceeds the sandbox message limit",
                "detector_output_invalid",
            ),
            "detector_output_invalid",
        )


def _build_video(message: dict[str, Any]) -> dict[str, Any]:
    frames_path = str(message["frames_path"])
    num_frames = int(message["num_frames"])
    width = int(message["width"])
    height = int(message["height"])
    frames = np.load(frames_path, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(
            f"staged frames must be uint8 (T,H,W,3), got {frames.dtype} {frames.shape}"
        )
    if tuple(frames.shape) != (num_frames, height, width, 3):
        raise ValueError(
            f"staged frames shape {tuple(frames.shape)} does not match the metadata"
        )
    return {
        "frames": frames,
        "frames_path": frames_path,
        "video_path": str(message["video_path"]),
        "fps": float(message["fps"]),
        "num_frames": num_frames,
        "width": width,
        "height": height,
        "duration": float(message["duration"]),
        "issue_types": [str(name) for name in message["issue_types"]],
    }


def _write_error(stream: Any, exc: BaseException, fallback_mode: str) -> None:
    # Only worker-classified failures carry their own mode; any other exception
    # gets the fallback, so a foreign exception's attributes are never trusted.
    if isinstance(exc, _WorkerFailure):
        mode, message = exc.failure_mode, str(exc)
    else:
        mode, message = fallback_mode, _describe(exc)
    write_message(
        stream,
        {"ok": False, "error": message[:_MAX_ERROR_CHARS], "failure_mode": mode},
        max_bytes=MAX_RESPONSE_BYTES,
    )


if __name__ == "__main__":
    main()
