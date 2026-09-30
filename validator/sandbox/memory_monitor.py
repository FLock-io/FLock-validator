from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
from typing import Callable, Optional

# Runtime memory enforcement for sandboxed workers.
#
# A worker runs untrusted code, so its memory budget cannot be enforced by
# statically auditing what it loads: data can be reconstructed at run time (from
# a bytes blob, a file, a foreign module, a dynamic lookup) in ways a static walk
# cannot see. A hard ceiling on the worker's actual memory footprint is the
# representation-agnostic bound. This module measures the worker's resident
# memory without third-party dependencies and kills it if it stays over budget.

MemorySampler = Callable[[int], Optional[int]]


def read_process_rss_bytes(pid: int) -> Optional[int]:
    """Resident set size of ``pid`` in bytes, or ``None`` if it can't be read.

    Dependency-free so the sandbox never relies on an optional package being
    installed. Threads share the process RSS, and the sandbox seccomp policy only
    permits thread creation (not child processes), so a single pid fully accounts
    for the worker's system memory.
    """
    system = platform.system()
    if system == "Linux":
        try:
            with open(f"/proc/{pid}/status", "r") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            return None
        return None
    if system == "Darwin":
        try:
            result = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (subprocess.SubprocessError, OSError):
            return None
        value = result.stdout.strip()
        try:
            return int(value) * 1024 if value else None
        except ValueError:
            return None
    return None


class MemoryMonitor:
    """Polls a worker process's resident memory and kills it if it exceeds the
    limit for ``breaches_before_kill`` consecutive samples.

    The consecutive-breach requirement avoids killing on a single transient
    reading; the limit itself is the authoritative ceiling. On a breach the whole
    process group is killed (the worker is started in its own session) and the
    breach is recorded so the proxy can report a memory failure rather than a
    generic pipe error.
    """

    def __init__(
        self,
        pid: int,
        limit_bytes: int,
        *,
        sampler: MemorySampler = read_process_rss_bytes,
        poll_interval_seconds: float = 0.5,
        breaches_before_kill: int = 2,
    ) -> None:
        self._pid = pid
        self._limit_bytes = limit_bytes
        self._sampler = sampler
        self._poll_interval_seconds = poll_interval_seconds
        self._breaches_before_kill = max(1, breaches_before_kill)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"sandbox-mem-{pid}", daemon=True
        )
        self._breached = False
        self._peak_bytes = 0
        self._observed_at_kill = 0

    @property
    def breached(self) -> bool:
        return self._breached

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    @property
    def observed_at_kill(self) -> int:
        return self._observed_at_kill

    @property
    def limit_bytes(self) -> int:
        return self._limit_bytes

    def start(self) -> None:
        # A non-positive limit means "unbounded"; skip monitoring entirely.
        if self._limit_bytes > 0:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        consecutive = 0
        while not self._stop.is_set():
            usage = self._sampler(self._pid)
            if usage is None:
                # The process is gone or its memory can't be read; nothing to
                # enforce.
                return
            if usage > self._peak_bytes:
                self._peak_bytes = usage
            if usage > self._limit_bytes:
                consecutive += 1
                if consecutive >= self._breaches_before_kill:
                    # Record the breach BEFORE killing so a proxy that observes
                    # the resulting EOF always sees it.
                    self._observed_at_kill = usage
                    self._breached = True
                    self._kill()
                    return
            else:
                consecutive = 0
            self._stop.wait(self._poll_interval_seconds)

    def _kill(self) -> None:
        # The worker runs in its own session (start_new_session=True), so its pid
        # is its process-group id; kill the whole group to catch any threads.
        try:
            os.killpg(self._pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                os.kill(self._pid, signal.SIGKILL)
            except OSError:
                pass
