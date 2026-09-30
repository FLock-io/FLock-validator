from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from validator.sandbox.errors import SandboxError, SandboxUnavailableError
from validator.sandbox.hardening import sandbox_environment
from validator.sandbox.memory_monitor import MemoryMonitor
from validator.sandbox.protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MessageTooLargeError,
    read_message,
    write_message,
)


class SandboxProcess:
    """A worker process running untrusted code, spoken to over length-prefixed JSON.

    The worker starts in its own session (so the whole group can be killed) with
    a clean environment, a private working directory, no inherited descriptors and
    stderr discarded. On Linux the worker applies Landlock + seccomp to itself;
    on macOS the whole command runs under ``sandbox-exec`` with networking denied.
    A host-side monitor bounds the worker's resident memory.

    Every failure that means the worker cannot continue kills it and raises a
    fatal ``SandboxError`` with one of: the caller's ``timeout_mode``,
    ``memory_exceeded``, ``crashed`` or ``protocol_error``.
    """

    def __init__(
        self,
        argv: Sequence[str],
        *,
        memory_limit_bytes: int,
        temp_prefix: str = "sandbox-worker-",
        unsafe_local_env_var: str | None = None,
        deny_read_paths: Sequence[Path] = (),
    ) -> None:
        # Set first so __del__/close are safe if construction fails part-way.
        self._closed = True
        self._monitor: MemoryMonitor | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._temp_dir = Path(tempfile.mkdtemp(prefix=temp_prefix))
        try:
            command = wrap_command(
                argv,
                unsafe_local_env_var=unsafe_local_env_var,
                deny_read_paths=deny_read_paths,
            )
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(self._temp_dir),
                env=sandbox_environment(self._temp_dir),
                close_fds=True,
                start_new_session=True,
            )
        except Exception:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            raise
        self._closed = False
        # The host-side ceiling on the live process is the authoritative memory
        # bound: however data is represented in the worker, it occupies memory.
        self._monitor = MemoryMonitor(self._process.pid, memory_limit_bytes)
        self._monitor.start()

    # -- public API -----------------------------------------------------------

    @property
    def temp_dir(self) -> Path:
        """The worker's private, writable working directory (its TMPDIR/HOME)."""
        return self._temp_dir

    @property
    def pid(self) -> int:
        return self._require_process().pid

    def send(self, message: dict[str, Any]) -> None:
        process = self._live_process()
        if process.stdin is None or process.poll() is not None:
            # If the monitor killed the worker for exceeding its memory budget,
            # report that rather than a generic exit, so a caller's retry loop
            # cannot mask the real cause on a later send.
            self._raise_if_memory_exceeded()
            raise SandboxError("Sandbox worker exited unexpectedly", "crashed")
        try:
            write_message(process.stdin, message, max_bytes=MAX_REQUEST_BYTES)
        except MessageTooLargeError as exc:
            raise SandboxError(
                f"Request exceeds the sandbox message limit: {exc}", "protocol_error"
            ) from exc
        except (BrokenPipeError, OSError) as exc:
            _kill_process_group(process)
            self._raise_if_memory_exceeded(exc)
            raise SandboxError(
                "Sandbox worker exited while receiving a request", "crashed"
            ) from exc

    def receive(self, timeout_seconds: float, timeout_mode: str) -> dict[str, Any]:
        process = self._live_process()
        if process.stdout is None:
            raise SandboxError("Sandbox protocol is unavailable", "protocol_error")
        fd = process.stdout.fileno()
        deadline = time.monotonic() + timeout_seconds
        try:
            return read_message(fd, deadline, max_bytes=MAX_RESPONSE_BYTES)
        except TimeoutError as exc:
            _kill_process_group(process)
            self._raise_if_memory_exceeded(exc)
            raise SandboxError(
                f"Sandbox worker exceeded its {timeout_seconds:g}s wall-time limit",
                timeout_mode,
            ) from exc
        except EOFError as exc:
            # The pipe closed: the worker exited (crash, rlimit kill, or our own
            # memory-monitor kill). A memory kill closes the pipe too, so check
            # for it before calling this a generic crash.
            _kill_process_group(process)
            self._raise_if_memory_exceeded(exc)
            raise SandboxError("Sandbox worker exited unexpectedly", "crashed") from exc
        except (
            MessageTooLargeError,
            UnicodeDecodeError,
            ValueError,
            OSError,
            RecursionError,
        ) as exc:
            # ValueError covers JSONDecodeError and non-object payloads;
            # RecursionError a hostile, deeply nested JSON reply (json.loads raises
            # it, and it must not escape as an infrastructure error).
            _kill_process_group(process)
            self._raise_if_memory_exceeded(exc)
            raise SandboxError(
                f"Sandbox worker returned an invalid protocol response: {exc}",
                "protocol_error",
            ) from exc

    def request(
        self, message: dict[str, Any], timeout_seconds: float, timeout_mode: str
    ) -> dict[str, Any]:
        self.send(message)
        return self.receive(timeout_seconds, timeout_mode)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._monitor is not None:
            self._monitor.stop()
        process = self._process
        try:
            if process is not None and process.poll() is None:
                try:
                    if process.stdin is not None:
                        write_message(
                            process.stdin, {"op": "close"}, max_bytes=MAX_REQUEST_BYTES
                        )
                    process.wait(timeout=1)
                except Exception:  # noqa: BLE001 - fall back to the hard kill
                    _kill_process_group(process)
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
            # The worker is already dead; cleanup failure must not replace a
            # validation result or the original sandbox error.
            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def __enter__(self) -> "SandboxProcess":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - never raise from a finaliser
            pass

    # -- internals ------------------------------------------------------------

    def _require_process(self) -> subprocess.Popen[bytes]:
        if self._process is None:
            raise SandboxError("Sandbox worker was never started", "crashed")
        return self._process

    def _live_process(self) -> subprocess.Popen[bytes]:
        if self._closed:
            raise SandboxError("Sandbox worker is closed", "crashed")
        return self._require_process()

    def _raise_if_memory_exceeded(self, cause: BaseException | None = None) -> None:
        monitor = self._monitor
        if monitor is None or not monitor.breached:
            return
        limit_gib = monitor.limit_bytes / 1024**3
        observed_gib = monitor.observed_at_kill / 1024**3
        raise SandboxError(
            f"Sandbox worker exceeded its {limit_gib:.2g} GiB memory limit "
            f"(observed {observed_gib:.2f} GiB)",
            "memory_exceeded",
        ) from cause


def wrap_command(
    argv: Sequence[str],
    *,
    unsafe_local_env_var: str | None = None,
    deny_read_paths: Sequence[Path] = (),
) -> list[str]:
    """Return the command that starts ``argv`` inside this platform's sandbox.

    ``deny_read_paths`` (e.g. the extracted validation package with its hidden
    labels, or a credentials file) are made unreadable on macOS, where the
    profile otherwise allows all file reads. On Linux, Landlock already grants
    the worker only an allowlist of paths.

    Raises ``SandboxUnavailableError`` when the host cannot sandbox at all.
    """
    base = list(argv)
    system = platform.system()
    if system == "Linux":
        # Landlock and seccomp are applied by the worker itself, before it
        # imports any untrusted code.
        return base
    if system == "Darwin":
        # Tests use trusted fixture code and may themselves run inside a host
        # sandbox that forbids nesting sandbox-exec. The worker still has a clean
        # environment, a separate process, and resource/time limits in this mode.
        if os.getenv("PYTEST_CURRENT_TEST") or (
            unsafe_local_env_var and os.getenv(unsafe_local_env_var) == "1"
        ):
            logger.debug("Running sandbox worker without sandbox-exec (test/local mode)")
            return base
        if shutil.which("sandbox-exec"):
            return ["sandbox-exec", "-p", darwin_profile(deny_read_paths), *base]
    raise SandboxUnavailableError(
        f"No supported network sandbox is available on {system or 'this platform'}"
    )


def darwin_profile(deny_read_paths: Sequence[Path] = ()) -> str:
    """sandbox-exec profile: no network, and no reads of ``deny_read_paths``."""
    rules = ["(version 1)", "(allow default)", "(deny network*)"]
    for path in deny_read_paths:
        # sandbox-exec matches real paths (/var -> /private/var on macOS).
        real = os.path.realpath(path)
        literal = '"' + real.replace("\\", "\\\\").replace('"', '\\"') + '"'
        rules.append(f"(deny file-read* (subpath {literal}))")
    return "".join(rules)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        # macOS reports EPERM for a group whose leader has exited but is not yet
        # reaped; either way there is nothing left to signal.
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
