from __future__ import annotations


class SandboxError(Exception):
    """A sandboxed worker failed in a way that is attributable to the worker.

    ``failure_mode`` is a short machine-readable tag. The generic sandbox raises
    ``memory_exceeded``, ``crashed`` and ``protocol_error`` itself, and passes
    through the ``timeout_mode`` the caller supplied for a wall-time breach. The
    task layer translates these into its own submission-error vocabulary.

    ``fatal`` is True when the worker process is gone (or was killed) and cannot
    serve further requests.
    """

    def __init__(self, message: str, failure_mode: str, *, fatal: bool = True):
        super().__init__(message)
        self.failure_mode = failure_mode
        self.fatal = fatal


class SandboxUnavailableError(Exception):
    """This host cannot provide the required sandbox.

    That is an infrastructure problem (no Landlock, no seccomp, no sandbox-exec,
    unsupported platform), not the submitter's fault, so callers must surface it
    as a recoverable/retryable condition rather than scoring the submission.
    """
