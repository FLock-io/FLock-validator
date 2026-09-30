"""Video inconsistency detection validation module.

The public names are resolved lazily (PEP 562) so lightweight consumers — the
sandbox worker, the dataset builder, the trainer sample scripts — can import
``issue_types`` / ``synthesis`` / ``manifest`` without pulling in the host-side
validation stack (Hugging Face Hub, the sandbox launcher, scoring).
"""

from __future__ import annotations

from typing import Any

_MODULE_EXPORTS = (
    "VideoInconsistencyConfig",
    "VideoInconsistencyInputData",
    "VideoInconsistencyMetrics",
    "VideoInconsistencyValidationModule",
    "MODULE",
)

__all__ = list(_MODULE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _MODULE_EXPORTS:
        from validator.modules.video_inconsistency import module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
