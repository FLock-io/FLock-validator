from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from validator.exceptions import RecoverableException
from validator.modules.video_inconsistency import detector as detector_module
from validator.modules.video_inconsistency.detector import (
    DEFAULT_ADAPTER_FILENAME,
    IsolatedDetector,
    load_detector_from_adapter,
    stage_clip,
    _write_new_file,
)
from validator.modules.video_inconsistency.errors import VideoSubmissionError
from validator.modules.video_inconsistency.issue_types import ISSUE_TYPE_NAMES
from validator.sandbox import (
    MAX_REQUEST_BYTES,
    MessageTooLargeError,
    SandboxError,
    SandboxProcess,
    SandboxUnavailableError,
    encode_message,
    read_message,
    read_process_rss_bytes,
    sandbox_environment,
    to_jsonable,
    wrap_command,
)
from validator.sandbox import process as sandbox_process_module

REPO_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


# --- helpers ------------------------------------------------------------------


def _write_adapter(root: Path, body: str, filename: str = DEFAULT_ADAPTER_FILENAME) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / filename).write_text(textwrap.dedent(body))
    return root


def _load(model_dir: Path, **overrides: Any) -> IsolatedDetector:
    kwargs: dict[str, Any] = dict(
        device="cpu",
        torch_dtype="float32",
        load_timeout_seconds=60,
        detect_timeout_seconds=30,
        memory_limit_bytes=16 * GIB,
        cpu_time_seconds=600,
    )
    kwargs.update(overrides)
    return load_detector_from_adapter(model_dir, DEFAULT_ADAPTER_FILENAME, **kwargs)


def _frames(num_frames: int = 12, height: int = 8, width: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)


def _video_file(tmp_path: Path, name: str = "clip_secret_id.mp4") -> Path:
    path = tmp_path / name
    path.write_bytes(b"not-really-an-mp4-but-only-copied")
    return path


def _detect(detector: IsolatedDetector, tmp_path: Path, frames: np.ndarray | None = None) -> Any:
    return detector.detect(
        frames=_frames() if frames is None else frames,
        video_path=_video_file(tmp_path),
        fps=4.0,
    )


def _fail_mode(excinfo: pytest.ExceptionInfo[VideoSubmissionError]) -> tuple[str, bool]:
    return excinfo.value.failure_mode, excinfo.value.fatal


_HAPPY_ADAPTER = """
import os
import numpy as np

class Detector:
    def detect(self, video):
        frames = video["frames"]
        return {
            "issues": [
                {
                    "type": "zoom_jump",
                    "start_time": np.float32(1.5),
                    "end_time": np.float64(2.5),
                    "confidence": np.float32(0.75),
                    "bbox": np.array([0.25, 0.5, 0.75, 1.0]),
                }
            ],
            "seen": {
                "shape": list(frames.shape),
                "dtype": str(frames.dtype),
                "is_memmap": isinstance(frames, np.memmap),
                "writeable": bool(frames.flags.writeable),
                "mean": float(frames.mean()),
                "fps": video["fps"],
                "duration": video["duration"],
                "num_frames": video["num_frames"],
                "width": video["width"],
                "height": video["height"],
                "issue_types": video["issue_types"],
                "video_path": video["video_path"],
                "frames_path": video["frames_path"],
                "video_bytes": len(open(video["video_path"], "rb").read()),
                "npy_shape": list(np.load(video["frames_path"]).shape),
                "keys": sorted(video),
            },
        }

def load_detector(model_dir, device, dtype):
    return Detector()
"""


# --- happy path and per-call behaviour -----------------------------------------


def test_happy_path_delivers_clip_and_returns_numpy_result(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    frames = _frames()
    video = _video_file(tmp_path)
    with _load(model_dir) as detector:
        result = detector.detect(frames=frames, video_path=video, fps=4.0)

    issue = result["issues"][0]
    assert issue == {
        "type": "zoom_jump",
        "start_time": 1.5,
        "end_time": 2.5,
        "confidence": 0.75,
        "bbox": [0.25, 0.5, 0.75, 1.0],
    }
    seen = result["seen"]
    assert seen["shape"] == [12, 8, 16, 3]
    assert seen["npy_shape"] == [12, 8, 16, 3]
    assert seen["dtype"] == "uint8"
    assert seen["is_memmap"] is True
    assert seen["writeable"] is False
    assert seen["mean"] == pytest.approx(float(frames.mean()))
    assert seen["fps"] == 4.0
    assert seen["duration"] == 3.0
    assert (seen["num_frames"], seen["width"], seen["height"]) == (12, 16, 8)
    assert seen["issue_types"] == list(ISSUE_TYPE_NAMES)
    assert os.path.basename(seen["video_path"]) == "input.mp4"
    assert os.path.basename(seen["frames_path"]) == "frames.npy"
    assert seen["video_bytes"] == video.stat().st_size
    assert sorted(seen["keys"]) == sorted(
        [
            "frames", "frames_path", "video_path", "fps", "num_frames",
            "width", "height", "duration", "issue_types",
        ]
    )
    # Nothing about the package's clip id may reach the trainer.
    assert "clip_secret_id" not in seen["video_path"]
    assert "clip_secret_id" not in seen["frames_path"]


def test_staged_clip_is_removed_after_each_detect(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    with _load(model_dir) as detector:
        _detect(detector, tmp_path)
        _detect(detector, tmp_path)
        assert list(detector._input_dir.iterdir()) == []


def test_detect_exception_is_non_fatal_and_detector_keeps_working(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Detector:
            def __init__(self):
                self.calls = 0

            def detect(self, video):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("boom " + "x" * 2000)
                return []

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_execution_failed", False)
        assert "RuntimeError: boom" in str(excinfo.value)
        assert len(str(excinfo.value)) <= 500
        assert _detect(detector, tmp_path) == []


def test_detect_calling_sys_exit_is_non_fatal(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import sys

        class Detector:
            def detect(self, video):
                sys.exit(3)

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_execution_failed", False)


def test_non_serialisable_result_is_non_fatal_output_invalid(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Detector:
            def __init__(self):
                self.calls = 0

            def detect(self, video):
                self.calls += 1
                if self.calls == 1:
                    return {"issues": [object()]}
                return {"issues": []}

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_output_invalid", False)
        assert _detect(detector, tmp_path) == {"issues": []}


def test_nan_result_is_output_invalid(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Detector:
            def detect(self, video):
                return {"issues": [{"type": "zoom_jump", "start_time": 0.0,
                                     "end_time": 1.0, "confidence": float("nan")}]}

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_output_invalid", False)


def test_oversized_result_is_output_invalid_and_non_fatal(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Detector:
            def __init__(self):
                self.calls = 0

            def detect(self, video):
                self.calls += 1
                if self.calls == 1:
                    return {"description": "x" * (5 * 1024 * 1024)}
                return []

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_output_invalid", False)
        assert _detect(detector, tmp_path) == []


def test_bare_list_result_passes_through(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Detector:
            def detect(self, video):
                return [{"type": "dropped_frames", "start_time": 1.0, "end_time": 1.0}]

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        assert _detect(detector, tmp_path) == [
            {"type": "dropped_frames", "start_time": 1.0, "end_time": 1.0}
        ]


# --- load-time failures ---------------------------------------------------------


def test_load_detector_raising_is_model_load_failed(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        def load_detector(model_dir, device, dtype):
            raise RuntimeError("weights are corrupt")
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("model_load_failed", True)
    assert "weights are corrupt" in str(excinfo.value)


def test_load_detector_is_retried_once(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import os

        class Detector:
            def detect(self, video):
                return []

        def load_detector(model_dir, device, dtype):
            marker = os.path.join(os.environ["TMPDIR"], "attempted")
            if not os.path.exists(marker):
                open(marker, "w").close()
                raise RuntimeError("flaky first attempt")
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        assert _detect(detector, tmp_path) == []


def test_missing_load_detector_is_adapter_contract(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", "x = 1\n")
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_contract", True)


def test_detector_without_detect_is_adapter_contract(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        def load_detector(model_dir, device, dtype):
            return object()
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_contract", True)


def test_syntax_error_is_adapter_import_failed(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", "def load_detector(:\n")
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_import_failed", True)


def test_import_time_exception_is_adapter_import_failed(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", "import definitely_not_a_module\n")
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_import_failed", True)


def test_trainer_cannot_forge_sandbox_unavailable(tmp_path: Path):
    # A trainer raising an exception that *claims* an infrastructure failure must
    # still be treated as its own load failure, never a retryable host problem.
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        class Forged(Exception):
            failure_mode = "sandbox_unavailable"

        def load_detector(model_dir, device, dtype):
            raise Forged("pretend the host is broken")
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("model_load_failed", True)


def test_missing_adapter_file_is_adapter_missing(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_missing", True)


def test_adapter_filename_escape_is_adapter_contract(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "x.py").write_text("def load_detector(**kw): pass\n")
    with pytest.raises(VideoSubmissionError) as excinfo:
        load_detector_from_adapter(
            model_dir, "../x.py", device="cpu", torch_dtype="float32",
            load_timeout_seconds=5, detect_timeout_seconds=5,
            memory_limit_bytes=GIB, cpu_time_seconds=60,
        )
    assert _fail_mode(excinfo) == ("adapter_contract", True)


def test_adapter_symlink_escape_is_rejected(tmp_path: Path):
    model_dir = tmp_path / "a" / "b" / "model"
    model_dir.mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("def load_detector(**kw): pass\n")
    (model_dir / DEFAULT_ADAPTER_FILENAME).symlink_to(outside)
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir)
    assert _fail_mode(excinfo) == ("adapter_symlink_escape", True)


def test_adapter_loads_through_hf_style_symlink(tmp_path: Path):
    repo = tmp_path / "models--org--repo"
    blobs = repo / "blobs"
    snapshot = repo / "snapshots" / "abc123"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    (blobs / "deadbeef").write_text(
        "class D:\n    def detect(self, video):\n        return []\n"
        "def load_detector(model_dir, device, dtype):\n    return D()\n"
    )
    (snapshot / DEFAULT_ADAPTER_FILENAME).symlink_to(blobs / "deadbeef")
    with _load(snapshot) as detector:
        assert _detect(detector, tmp_path) == []


def test_slow_load_hits_model_load_timeout(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import time

        def load_detector(model_dir, device, dtype):
            time.sleep(60)
        """,
    )
    started = time.monotonic()
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir, load_timeout_seconds=1)
    assert _fail_mode(excinfo) == ("model_load_timeout", True)
    assert time.monotonic() - started < 20


# --- wall-time, memory, crash ---------------------------------------------------


def test_detect_timeout_is_fatal_and_detector_is_dead_afterwards(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import time

        class Detector:
            def detect(self, video):
                time.sleep(60)

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    detector = _load(model_dir, detect_timeout_seconds=1)
    try:
        started = time.monotonic()
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_timeout", True)
        assert time.monotonic() - started < 20
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_crashed", True)
    finally:
        detector.close()


def test_worker_crash_during_detect_is_detector_crashed(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import os

        class Detector:
            def detect(self, video):
                os._exit(7)

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    detector = _load(model_dir)
    try:
        with pytest.raises(VideoSubmissionError) as excinfo:
            _detect(detector, tmp_path)
        assert _fail_mode(excinfo) == ("detector_crashed", True)
    finally:
        detector.close()


@pytest.mark.skipif(
    read_process_rss_bytes(os.getpid()) is None,
    reason="this platform cannot measure process RSS",
)
def test_memory_blow_up_is_detector_memory_exceeded(tmp_path: Path):
    # ~300 MB of touched (resident) memory against a 200 MB budget. On macOS the
    # host memory monitor kills the worker. On Linux/CPU the kernel RLIMIT_AS
    # backstop may stop the allocation first, which surfaces as a failed load.
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import time
        import numpy as np

        class Detector:
            def detect(self, video):
                return []

        def load_detector(model_dir, device, dtype):
            hog = np.ones(300_000_000, dtype=np.uint8)
            # Keep every page hot so the OS cannot compress or page it out of RSS.
            deadline = time.time() + 8
            while time.time() < deadline:
                hog[::4096] += 1
                time.sleep(0.05)
            detector = Detector()
            detector.hog = hog
            return detector
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _load(model_dir, memory_limit_bytes=200 * 1024 * 1024)
    mode = excinfo.value.failure_mode
    if sys.platform == "darwin":
        assert mode == "detector_memory_exceeded"
    else:
        assert mode in {"detector_memory_exceeded", "model_load_failed"}
    assert excinfo.value.fatal is True


# --- isolation of the environment and filesystem --------------------------------


def test_worker_environment_has_no_validator_secrets(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("FLOCK_API_KEY", "must-not-cross-process-boundary")
    monkeypatch.setenv("HF_TOKEN", "must-not-cross-process-boundary")
    monkeypatch.setenv("SOME_OTHER_SECRET", "must-not-cross-process-boundary")
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import os

        class Detector:
            def detect(self, video):
                return {"env": dict(os.environ)}

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        env = _detect(detector, tmp_path)["env"]
    assert "FLOCK_API_KEY" not in env
    assert "HF_TOKEN" not in env
    assert "SOME_OTHER_SECRET" not in env
    assert not any("must-not-cross" in value for value in env.values())
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert env["HF_DATASETS_OFFLINE"] == "1"


def test_sandbox_environment_is_an_allowlist(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    env = sandbox_environment(tmp_path)
    assert "HF_TOKEN" not in env
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["HOME"] == str(tmp_path) and env["TMPDIR"] == str(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"


def test_parameter_count_telemetry_for_torch_module(tmp_path: Path):
    pytest.importorskip("torch")
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import torch

        class Detector:
            def __init__(self):
                self.net = torch.nn.Linear(3, 4)
                self.alias = self.net

            def detect(self, video):
                return []

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    with _load(model_dir) as detector:
        assert detector.parameter_count == 3 * 4 + 4


def test_parameter_count_is_none_without_torch_modules(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    with _load(model_dir) as detector:
        assert detector.parameter_count is None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


@pytest.mark.skipif(
    sys.platform != "linux" or not _cuda_available(), reason="needs Linux with a CUDA GPU"
)
def test_worker_runs_cuda_inside_the_sandbox_within_the_memory_cap(tmp_path: Path):
    model_dir = _write_adapter(
        tmp_path / "model",
        """
        import torch

        class Detector:
            def __init__(self, device):
                self.device = device
                self.net = torch.nn.Linear(64, 64).to(device)

            def detect(self, video):
                if video["num_frames"] > 1000:
                    torch.empty(int(8 * 1024**3), dtype=torch.uint8, device=self.device)
                x = torch.from_numpy(video["frames"][:4].copy()).to(self.device).float()
                return {"issues": [], "device": torch.cuda.get_device_name(0),
                        "sum": float(x.mean())}

        def load_detector(model_dir, device, dtype):
            return Detector(device)
        """,
    )
    detector = load_detector_from_adapter(
        model_dir,
        "flock_video_adapter.py",
        device="cuda",
        torch_dtype="bfloat16",
        load_timeout_seconds=300,
        detect_timeout_seconds=120,
        memory_limit_bytes=4 * 1024**3,
        cpu_time_seconds=600,
    )
    with detector:
        result = detector.detect(frames=_frames(), video_path=_video_file(tmp_path), fps=15.0)
        assert result["device"]
        with pytest.raises(VideoSubmissionError):
            # 8 GiB on a 4 GiB cap: the CUDA allocator fraction refuses it.
            detector.detect(frames=_frames(1001), video_path=_video_file(tmp_path), fps=15.0)

    oversize = _write_adapter(
        tmp_path / "oversize",
        """
        import torch

        class Detector:
            def detect(self, video):
                return []

        def load_detector(model_dir, device, dtype):
            d = Detector()
            d.big = torch.empty(int(8 * 1024**3), dtype=torch.uint8, device=device)
            return d
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        load_detector_from_adapter(
            oversize,
            "flock_video_adapter.py",
            device="cuda",
            torch_dtype="bfloat16",
            load_timeout_seconds=300,
            detect_timeout_seconds=120,
            memory_limit_bytes=4 * 1024**3,
            cpu_time_seconds=600,
        )
    assert excinfo.value.failure_mode in {"model_load_failed", "detector_memory_exceeded"}


@pytest.mark.skipif(sys.platform != "linux", reason="Linux Landlock/seccomp behaviour")
def test_worker_denies_network_filesystem_and_process_escape(tmp_path: Path):
    secret = tmp_path / "validator-secret.txt"
    secret.write_text("must-not-be-readable")
    # A live host daemon socket (stands in for e.g. /var/run/docker.sock). Landlock
    # does not mediate UNIX connect, so seccomp must refuse it.
    import socket as host_socket

    daemon_path = tmp_path / "host-daemon.sock"
    daemon = host_socket.socket(host_socket.AF_UNIX, host_socket.SOCK_STREAM)
    daemon.bind(str(daemon_path))
    daemon.listen(1)
    abstract = host_socket.socket(host_socket.AF_UNIX, host_socket.SOCK_STREAM)
    abstract.bind("\0flock-test-abstract-daemon")
    abstract.listen(1)
    model_path = tmp_path / "model"
    model_dir = _write_adapter(
        model_path,
        f"""
        import os
        import socket
        import subprocess

        class Detector:
            def detect(self, video):
                denied = {{}}
                def attempt(name, fn):
                    try:
                        fn()
                    except OSError:
                        denied[name] = True
                    else:
                        denied[name] = False
                def unix_connect(address):
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(address)
                def unix_sendto():
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                    s.sendto(b"x", {str(daemon_path)!r})
                attempt("socket", socket.socket)
                attempt("socket_inet6", lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM))
                attempt("socket_netlink", lambda: socket.socket(socket.AF_NETLINK, socket.SOCK_RAW))
                attempt("unix_connect_path", lambda: unix_connect({str(daemon_path)!r}))
                attempt("unix_connect_abstract", lambda: unix_connect("\\0flock-test-abstract-daemon"))
                attempt("unix_sendto", unix_sendto)
                attempt("socketpair", socket.socketpair)
                # allowed: a bare local socket (the CUDA driver creates one)
                socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET).close()
                # only the worker's own /proc entries are granted, never another process's
                attempt("parent_proc", lambda: open("/proc/%d/cmdline" % os.getppid()).read())
                attempt("secret", lambda: open({str(secret)!r}).read())
                attempt("exec", lambda: subprocess.run(["/bin/true"], check=True))
                attempt("repo", lambda: os.listdir({str(REPO_ROOT)!r}))
                attempt("write_input", lambda: open(
                    os.path.join(os.path.dirname(video["frames_path"]), "planted"), "w"))
                attempt("write_model", lambda: open(
                    os.path.join(model_dir, "planted"), "w"))
                # allowed: private temp dir and reading the staged frames
                open(os.path.join(os.environ["TMPDIR"], "ok"), "w").write("ok")
                denied["read_frames"] = bool(video["frames"].shape[0])
                return denied

        model_dir = {str(model_path)!r}

        def load_detector(model_dir, device, dtype):
            return Detector()
        """,
    )
    try:
        detector = _load(model_dir)
    except RecoverableException as exc:
        pytest.skip(f"host cannot provide Landlock/seccomp: {exc}")
    try:
        with detector:
            outcome = _detect(detector, tmp_path)
    finally:
        daemon.close()
        abstract.close()
    assert outcome == {
        "socket": True,
        "socket_inet6": True,
        "socket_netlink": True,
        "unix_connect_path": True,
        "unix_connect_abstract": True,
        "unix_sendto": True,
        "socketpair": True,
        "parent_proc": True,
        "secret": True,
        "exec": True,
        "repo": True,
        "write_input": True,
        "write_model": True,
        "read_frames": True,
    }


# --- staging security -----------------------------------------------------------


def test_stage_clip_layout_and_content(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    frames = _frames()
    video = _video_file(tmp_path)
    clip_dir = stage_clip(input_dir, frames, video)
    assert clip_dir.parent == input_dir
    assert len(clip_dir.name) == 32 and "clip_secret_id" not in clip_dir.name
    assert sorted(p.name for p in clip_dir.iterdir()) == ["frames.npy", "input.mp4"]
    assert np.array_equal(np.load(clip_dir / "frames.npy"), frames)
    assert (clip_dir / "input.mp4").read_bytes() == video.read_bytes()
    assert (clip_dir.stat().st_mode & 0o777) == 0o700


def test_stage_clip_uses_unpredictable_directory_names(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    names = {
        stage_clip(input_dir, _frames(2), _video_file(tmp_path)).name for _ in range(5)
    }
    assert len(names) == 5


def test_stage_clip_refuses_planted_directory_symlink(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    redirected = tmp_path / "elsewhere"
    redirected.mkdir()
    monkeypatch.setattr(detector_module.secrets, "token_hex", lambda _n: "fixed")
    (input_dir / "fixed").symlink_to(redirected)
    with pytest.raises(FileExistsError):
        stage_clip(input_dir, _frames(2), _video_file(tmp_path))
    assert list(redirected.iterdir()) == []


def test_write_new_file_refuses_planted_symlinks(tmp_path: Path):
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")
    live_link = tmp_path / "live-link"
    live_link.symlink_to(victim)
    dangling = tmp_path / "dangling-link"
    dangling_target = tmp_path / "does-not-exist"
    dangling.symlink_to(dangling_target)

    for link in (live_link, dangling):
        with pytest.raises(FileExistsError):
            _write_new_file(link, lambda handle: handle.write(b"overwritten"))
    assert victim.read_text() == "keep me"
    assert not dangling_target.exists()


def test_write_new_file_refuses_existing_file(tmp_path: Path):
    existing = tmp_path / "already"
    existing.write_text("original")
    with pytest.raises(FileExistsError):
        _write_new_file(existing, lambda handle: handle.write(b"new"))
    assert existing.read_text() == "original"


def test_stage_clip_rejects_bad_frames(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    video = _video_file(tmp_path)
    with pytest.raises(ValueError):
        stage_clip(input_dir, _frames().astype(np.float32), video)
    with pytest.raises(ValueError):
        stage_clip(input_dir, _frames()[..., :2], video)
    with pytest.raises(ValueError):
        stage_clip(input_dir, np.asfortranarray(_frames()), video)
    assert list(input_dir.iterdir()) == []


def test_stage_clip_cleans_up_when_video_is_missing(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        stage_clip(input_dir, _frames(2), tmp_path / "missing.mp4")
    assert list(input_dir.iterdir()) == []


# --- lifecycle -------------------------------------------------------------------


def test_close_is_idempotent_and_removes_directories(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    detector = _load(model_dir)
    input_dir = detector._input_dir
    worker_temp = detector._process.temp_dir
    assert input_dir.exists() and worker_temp.exists()
    detector.close()
    detector.close()
    assert not input_dir.exists() and not worker_temp.exists()
    with pytest.raises(VideoSubmissionError) as excinfo:
        _detect(detector, tmp_path)
    assert _fail_mode(excinfo) == ("detector_crashed", True)


def test_limits_must_be_positive(tmp_path: Path):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    with pytest.raises(ValueError):
        _load(model_dir, detect_timeout_seconds=0)


def test_host_without_sandbox_is_recoverable_not_a_submission_error(
    tmp_path: Path, monkeypatch
):
    model_dir = _write_adapter(tmp_path / "model", _HAPPY_ADAPTER)
    monkeypatch.setattr(
        sandbox_process_module, "platform", SimpleNamespace(system=lambda: "Plan9")
    )
    with pytest.raises(RecoverableException):
        _load(model_dir)


def test_wrap_command_platform_rules(monkeypatch):
    def fake_platform(name: str) -> SimpleNamespace:
        return SimpleNamespace(system=lambda: name)

    argv = ["python", "worker.py"]
    monkeypatch.setattr(sandbox_process_module, "platform", fake_platform("Linux"))
    assert wrap_command(argv) == argv

    monkeypatch.setattr(sandbox_process_module, "platform", fake_platform("Darwin"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("MY_UNSAFE_FLAG", raising=False)
    monkeypatch.setattr(sandbox_process_module.shutil, "which", lambda _n: None)
    with pytest.raises(SandboxUnavailableError):
        wrap_command(argv, unsafe_local_env_var="MY_UNSAFE_FLAG")

    monkeypatch.setenv("MY_UNSAFE_FLAG", "1")
    assert wrap_command(argv, unsafe_local_env_var="MY_UNSAFE_FLAG") == argv

    monkeypatch.delenv("MY_UNSAFE_FLAG")
    monkeypatch.setattr(
        sandbox_process_module.shutil, "which", lambda _n: "/usr/bin/sandbox-exec"
    )
    wrapped = wrap_command(argv, unsafe_local_env_var="MY_UNSAFE_FLAG")
    assert wrapped[0] == "sandbox-exec" and "(deny network*)" in wrapped[2]
    assert wrapped[3:] == argv

    monkeypatch.setattr(sandbox_process_module, "platform", fake_platform("Windows"))
    with pytest.raises(SandboxUnavailableError):
        wrap_command(argv)


# --- forged / faulty worker protocol ---------------------------------------------

_FAKE_WORKER_HEADER = f"""
import os
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from validator.sandbox.protocol import (
    HEADER, MAX_REQUEST_BYTES, MAX_RESPONSE_BYTES, read_message_blocking, write_message,
)

inp = os.fdopen(os.dup(0), "rb", buffering=0)
out = os.fdopen(os.dup(1), "wb", buffering=0)

def send(message):
    write_message(out, message, max_bytes=MAX_RESPONSE_BYTES)

def recv():
    return read_message_blocking(inp, max_bytes=MAX_REQUEST_BYTES)
"""


def _fake_worker(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER_HEADER + textwrap.dedent(body))
    return [sys.executable, "-u", str(script)]


def _start_with_fake_worker(
    tmp_path: Path, monkeypatch, body: str, **overrides: Any
) -> IsolatedDetector:
    argv = _fake_worker(tmp_path, body)
    monkeypatch.setattr(detector_module, "_worker_command", lambda **_kw: argv)
    kwargs: dict[str, Any] = dict(
        model_dir=tmp_path,
        adapter_filename=DEFAULT_ADAPTER_FILENAME,
        device="cpu",
        torch_dtype="float32",
        load_timeout_seconds=30,
        detect_timeout_seconds=10,
        memory_limit_bytes=16 * GIB,
        cpu_time_seconds=600,
    )
    kwargs.update(overrides)
    return IsolatedDetector.start(**kwargs)


def test_hardening_failure_reply_is_recoverable(tmp_path: Path, monkeypatch):
    with pytest.raises(RecoverableException, match="no landlock"):
        _start_with_fake_worker(
            tmp_path,
            monkeypatch,
            """
            send({"ok": False, "failure_mode": "sandbox_unavailable", "error": "no landlock"})
            """,
        )


def test_forged_sandbox_unavailable_during_detect_is_a_fatal_protocol_error(
    tmp_path: Path, monkeypatch
):
    detector = _start_with_fake_worker(
        tmp_path,
        monkeypatch,
        """
        send({"ok": True, "parameter_count": 5})
        recv()
        send({"ok": False, "failure_mode": "sandbox_unavailable", "error": "forged"})
        """,
    )
    assert detector.parameter_count == 5
    with pytest.raises(VideoSubmissionError) as excinfo:
        _detect(detector, tmp_path)
    assert _fail_mode(excinfo) == ("detector_protocol_error", True)


def test_unknown_worker_failure_mode_is_not_surfaced(tmp_path: Path, monkeypatch):
    with pytest.raises(VideoSubmissionError) as excinfo:
        _start_with_fake_worker(
            tmp_path,
            monkeypatch,
            """
            send({"ok": False, "failure_mode": "made_up_mode", "error": "x"})
            """,
        )
    assert _fail_mode(excinfo) == ("model_load_failed", True)


def test_garbage_response_is_detector_protocol_error(tmp_path: Path, monkeypatch):
    detector = _start_with_fake_worker(
        tmp_path,
        monkeypatch,
        """
        send({"ok": True})
        recv()
        payload = b"this is not json"
        out.write(HEADER.pack(len(payload)) + payload)
        import time; time.sleep(30)
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _detect(detector, tmp_path)
    assert _fail_mode(excinfo) == ("detector_protocol_error", True)


def test_oversized_response_header_is_detector_protocol_error(
    tmp_path: Path, monkeypatch
):
    detector = _start_with_fake_worker(
        tmp_path,
        monkeypatch,
        """
        send({"ok": True})
        recv()
        out.write(HEADER.pack(1 << 40))
        import time; time.sleep(30)
        """,
    )
    with pytest.raises(VideoSubmissionError) as excinfo:
        _detect(detector, tmp_path)
    assert _fail_mode(excinfo) == ("detector_protocol_error", True)


def test_worker_that_dies_while_loading_is_model_load_failed(
    tmp_path: Path, monkeypatch
):
    with pytest.raises(VideoSubmissionError) as excinfo:
        _start_with_fake_worker(tmp_path, monkeypatch, "os._exit(1)\n")
    assert _fail_mode(excinfo) == ("model_load_failed", True)


def test_load_failure_removes_staging_directory(tmp_path: Path, monkeypatch):
    created: list[Path] = []
    real_mkdtemp = detector_module.tempfile.mkdtemp

    def recording_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(detector_module.tempfile, "mkdtemp", recording_mkdtemp)
    with pytest.raises(VideoSubmissionError):
        _start_with_fake_worker(tmp_path, monkeypatch, "os._exit(1)\n")
    # The staging directory and the worker's private temp directory.
    assert {p.name.split("-")[2] for p in created} == {"input", "worker"}
    assert not any(p.exists() for p in created)


# --- generic SandboxProcess ---------------------------------------------------------


def test_sandbox_process_round_trip_and_idempotent_close(tmp_path: Path):
    argv = _fake_worker(
        tmp_path,
        """
        while True:
            message = recv()
            if message.get("op") == "close":
                break
            send({"echo": message})
        """,
    )
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        assert sandbox.temp_dir.is_dir()
        reply = sandbox.request({"op": "x", "n": 1}, 10, "some_timeout")
        assert reply == {"echo": {"op": "x", "n": 1}}
        assert sandbox.pid > 0
        temp_dir = sandbox.temp_dir
    sandbox.close()
    assert not temp_dir.exists()
    with pytest.raises(SandboxError) as excinfo:
        sandbox.send({"op": "x"})
    assert excinfo.value.failure_mode == "crashed"


def test_sandbox_process_deeply_nested_reply_is_a_protocol_error(tmp_path: Path):
    # json.loads raises RecursionError (not ValueError) on deep nesting; it must be
    # classified as the worker's protocol violation, never escape as an infra error.
    argv = _fake_worker(
        tmp_path,
        """
        payload = b"[" * 200000 + b"]" * 200000
        out.write(HEADER.pack(len(payload)) + payload)
        import time; time.sleep(30)
        """,
    )
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        with pytest.raises(SandboxError) as excinfo:
            sandbox.receive(20, "my_timeout")
    assert excinfo.value.failure_mode == "protocol_error"
    assert excinfo.value.fatal is True


def test_resolve_model_dir_refuses_host_paths_unless_local_is_allowed(tmp_path: Path):
    local = tmp_path / "submission"
    local.mkdir()
    assert detector_module.resolve_model_dir(str(local), allow_local=True) == local.resolve()
    # In production a trainer's reference is always a Hub id: an existing host path
    # is not a valid repo id and is rejected without touching the filesystem.
    with pytest.raises(VideoSubmissionError) as excinfo:
        detector_module.resolve_model_dir(str(local))
    assert excinfo.value.failure_mode == "model_reference_invalid"


def test_darwin_profile_denies_reading_the_given_paths(tmp_path: Path):
    from validator.sandbox.process import darwin_profile

    labels = tmp_path / 'pkg "quoted"'
    profile = darwin_profile([labels])
    assert profile.startswith("(version 1)(allow default)(deny network*)")
    real = os.path.realpath(labels).replace('"', '\\"')
    assert f'(deny file-read* (subpath "{real}"))' in profile
    assert darwin_profile() == "(version 1)(allow default)(deny network*)"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox-exec behaviour")
def test_sandbox_exec_profile_blocks_reading_denied_paths(tmp_path: Path):
    import shutil

    from validator.sandbox.process import darwin_profile

    if not shutil.which("sandbox-exec"):
        pytest.skip("sandbox-exec not available")
    secret_dir = tmp_path / "package"
    secret_dir.mkdir()
    (secret_dir / "manifest.json").write_text("hidden labels")
    allowed = tmp_path / "allowed.txt"
    allowed.write_text("ok")
    code = (
        "import sys\n"
        f"print(open({str(allowed)!r}).read())\n"
        f"try:\n    open({str(secret_dir / 'manifest.json')!r}).read()\n    print('LEAK')\n"
        "except OSError:\n    print('DENIED')\n"
    )
    completed = subprocess.run(
        ["sandbox-exec", "-p", darwin_profile([secret_dir]), sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0 and "sandbox_apply" in completed.stderr:
        pytest.skip("this host forbids nesting sandbox-exec")
    assert completed.stdout.split() == ["ok", "DENIED"]


def test_sandbox_process_timeout_uses_caller_mode(tmp_path: Path):
    argv = _fake_worker(tmp_path, "import time; time.sleep(60)\n")
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        with pytest.raises(SandboxError) as excinfo:
            sandbox.receive(0.5, "my_timeout")
    assert excinfo.value.failure_mode == "my_timeout"
    assert excinfo.value.fatal is True


def test_sandbox_process_reports_exit_as_crashed(tmp_path: Path):
    argv = _fake_worker(tmp_path, "os._exit(3)\n")
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        with pytest.raises(SandboxError) as excinfo:
            sandbox.receive(10, "t")
        assert excinfo.value.failure_mode == "crashed"
        # Sending to a dead worker must not hang or raise a raw OSError.
        with pytest.raises(SandboxError) as excinfo:
            sandbox.send({"op": "x"})
        assert excinfo.value.failure_mode == "crashed"


def test_sandbox_process_rejects_oversized_request(tmp_path: Path):
    argv = _fake_worker(tmp_path, "import time; time.sleep(60)\n")
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        with pytest.raises(SandboxError) as excinfo:
            sandbox.send({"op": "x", "blob": "y" * (MAX_REQUEST_BYTES + 1)})
    assert excinfo.value.failure_mode == "protocol_error"


def test_sandbox_process_memory_breach_wins_over_generic_exit(tmp_path: Path):
    # Use a sampler-free path: a real allocation is exercised in the detector
    # test above; here we only check that a recorded breach is reported on every
    # failure path, so a retry can never mask it as a plain crash.
    argv = _fake_worker(tmp_path, "import time; time.sleep(60)\n")
    with SandboxProcess(argv, memory_limit_bytes=16 * GIB) as sandbox:
        monitor = sandbox._monitor
        assert monitor is not None
        monitor._breached = True
        monitor._observed_at_kill = 5 * GIB
        os_kill = subprocess.run(["kill", "-9", str(sandbox.pid)], check=False)
        assert os_kill.returncode == 0
        with pytest.raises(SandboxError) as excinfo:
            sandbox.receive(10, "t")
        assert excinfo.value.failure_mode == "memory_exceeded"
        with pytest.raises(SandboxError) as excinfo:
            sandbox.send({"op": "x"})
        assert excinfo.value.failure_mode == "memory_exceeded"


# --- protocol ---------------------------------------------------------------------


def test_to_jsonable_converts_numpy_and_containers():
    value = {
        "a": np.float32(0.5),
        "b": np.int64(3),
        "c": np.bool_(True),
        "d": np.array([[1, 2], [3, 4]], dtype=np.int32),
        "e": (1, 2.5, "x", None),
        "f": [np.str_("s"), {"g": np.array([0.25])}],
    }
    assert to_jsonable(value) == {
        "a": 0.5,
        "b": 3,
        "c": True,
        "d": [[1, 2], [3, 4]],
        "e": [1, 2.5, "x", None],
        "f": ["s", {"g": [0.25]}],
    }
    converted = to_jsonable(value)
    assert type(converted["a"]) is float and type(converted["b"]) is int
    assert type(converted["c"]) is bool


@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
        np.float32("nan"),
        np.array([1.0, np.nan]),
        {"x": [float("inf")]},
        {1: "non-string key"},
        {"x": object()},
        {"x"},
        b"bytes",
        complex(1, 2),
        lambda: None,
    ],
)
def test_to_jsonable_rejects_unsupported_values(bad: Any):
    with pytest.raises(TypeError):
        to_jsonable(bad)


def test_to_jsonable_rejects_cycles():
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(TypeError):
        to_jsonable(cyclic)


def test_encode_message_enforces_size_cap_and_rejects_nan():
    with pytest.raises(MessageTooLargeError):
        encode_message({"x": "y" * 100}, max_bytes=50)
    with pytest.raises(ValueError):
        encode_message({"x": float("nan")}, max_bytes=1000)


def test_read_message_times_out_and_rejects_oversize():
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(TimeoutError):
            read_message(read_fd, time.monotonic() + 0.1, max_bytes=100)
        os.write(write_fd, encode_message({"k": 1}, max_bytes=100))
        assert read_message(read_fd, time.monotonic() + 1, max_bytes=100) == {"k": 1}
        os.write(write_fd, encode_message({"k": "z" * 200}, max_bytes=1000))
        with pytest.raises(MessageTooLargeError):
            read_message(read_fd, time.monotonic() + 1, max_bytes=100)
    finally:
        os.close(write_fd)
        os.close(read_fd)


def test_read_message_detects_eof():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    try:
        with pytest.raises(EOFError):
            read_message(read_fd, time.monotonic() + 1, max_bytes=100)
    finally:
        os.close(read_fd)
