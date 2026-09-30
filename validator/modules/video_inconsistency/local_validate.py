from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from validator.config import load_config_for_task
from validator.modules.video_inconsistency.detector import DEFAULT_ADAPTER_FILENAME
from validator.modules.video_inconsistency.module import (
    VideoInconsistencyConfig,
    VideoInconsistencyInputData,
    VideoInconsistencyValidationModule,
)


def run_local_validation(
    hg_repo_id: str,
    revision: str = "main",
    validation_data_url: str | None = None,
    adapter_filename: str = DEFAULT_ADAPTER_FILENAME,
    max_clips: int | None = None,
    device: str | None = None,
    torch_dtype: str | None = None,
    output_json: str | None = None,
    hf_token: str | None = None,
    config_dir: str = "configs",
) -> dict[str, Any]:
    """Validate one submission (HF repo id or local directory) without FedLedger."""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    config = load_config_for_task(
        task_id="local_video_inconsistency",
        task_type="video_inconsistency",
        config_model=VideoInconsistencyConfig,
        config_dir=config_dir,
    )
    # Local validation may point at a submission folder on disk; production
    # references always resolve through the Hugging Face Hub.
    config_updates: dict[str, Any] = {"allow_local_model_dir": True}
    if max_clips is not None:
        config_updates["max_clips"] = max_clips
    if device is not None:
        config_updates["device"] = device
    if torch_dtype is not None:
        config_updates["torch_dtype"] = torch_dtype
    if config_updates:
        config = config.model_copy(update=config_updates)

    data = VideoInconsistencyInputData(
        hg_repo_id=hg_repo_id,
        revision=revision,
        validation_data_url=validation_data_url,
        adapter_filename=adapter_filename,
    )
    module = VideoInconsistencyValidationModule(config=config)
    metrics = module.validate(data)
    result = metrics.model_dump()
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
    return result
