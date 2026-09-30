"""Generic, task-agnostic sandbox for running untrusted worker code.

A task supplies its own worker script (which calls the ``hardening`` functions on
startup, before importing untrusted code) and drives it through
``SandboxProcess``. See ``validator/modules/video_inconsistency`` for a client.
"""

from __future__ import annotations

from validator.sandbox.errors import SandboxError, SandboxUnavailableError
from validator.sandbox.hardening import (
    apply_cuda_memory_limit,
    apply_resource_limits,
    install_landlock,
    install_seccomp,
    protect_parent_secrets,
    sandbox_environment,
)
from validator.sandbox.memory_monitor import (
    MemoryMonitor,
    MemorySampler,
    read_process_rss_bytes,
)
from validator.sandbox.process import SandboxProcess, wrap_command
from validator.sandbox.protocol import (
    HEADER,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MessageTooLargeError,
    encode_message,
    read_exact,
    read_message,
    read_message_blocking,
    to_jsonable,
    write_message,
)

__all__ = [
    "HEADER",
    "MAX_REQUEST_BYTES",
    "MAX_RESPONSE_BYTES",
    "MemoryMonitor",
    "MemorySampler",
    "MessageTooLargeError",
    "SandboxError",
    "SandboxProcess",
    "SandboxUnavailableError",
    "apply_cuda_memory_limit",
    "apply_resource_limits",
    "encode_message",
    "install_landlock",
    "install_seccomp",
    "protect_parent_secrets",
    "read_exact",
    "read_message",
    "read_message_blocking",
    "read_process_rss_bytes",
    "sandbox_environment",
    "to_jsonable",
    "wrap_command",
    "write_message",
]
