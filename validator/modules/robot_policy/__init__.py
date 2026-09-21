from pathlib import Path
from typing import Literal
import uuid

from pydantic import ConfigDict, Field, field_validator
from validator.exceptions import RecoverableException
from validator.modules.base import BaseConfig, BaseInputData, BaseMetrics, BaseValidationModule

from .evaluation import benchmark_config, classify_failure, evaluate_assignment
from .suite import SuiteError, validate_suite_url


class RobotPolicyConfig(BaseConfig):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    task_config: str
    output_dir: str = ".cache/robot_policy"
    scoring_spec: str | None = None
    device: str = "cuda:0"
    batch: int = Field(default=1, ge=1)
    build_seed: int = Field(default=0, ge=0, le=2**32 - 1)
    onnx_provider: Literal["cpu", "cuda"] = "cpu"


class RobotPolicyInputData(BaseInputData):
    model_config = ConfigDict(frozen=True, extra="ignore", strict=True, hide_input_in_errors=True)
    hg_repo_id: str = Field(min_length=1)
    model_filename: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    max_params: int = Field(gt=0)
    validation_set_url: str = Field(min_length=1, repr=False)

    @field_validator("validation_set_url")
    @classmethod
    def valid_dataset_url(cls, value: str) -> str:
        try:
            return validate_suite_url(value)
        except SuiteError:
            raise ValueError("validation_set_url must be an allowed HTTPS URL") from None


class RobotPolicyMetrics(BaseMetrics):
    model_config = ConfigDict(frozen=True, extra="allow", allow_inf_nan=False)
    average_reward: float


def _check_runtime(task_id):
    # Register the upstream tasks before asking mjlab for their configuration.
    import mjlab_microduck.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    load_env_cfg(task_id)


class RobotPolicyValidationModule(BaseValidationModule):
    config_schema = RobotPolicyConfig
    input_data_schema = RobotPolicyInputData
    metrics_schema = RobotPolicyMetrics
    task_type = "robot_policy"

    def __init__(self, config: RobotPolicyConfig, **kwargs):
        self.config = config
        try:
            _, task, _ = benchmark_config(config.model_dump())
            _check_runtime(task["mjlab_task_id"])
        except Exception as exc:
            raise RecoverableException(f"RobotPolicy configuration or runtime unavailable: {exc}") from exc

    def validate(self, data: RobotPolicyInputData, **kwargs) -> RobotPolicyMetrics:
        run_id = uuid.uuid4().hex
        out = Path(self.config.output_dir) / run_id
        try:
            metrics, _ = evaluate_assignment(
                data.model_dump(), self.config.model_dump(), out,
                "flock-robot_policy-validator")
            return self.metrics_schema.model_validate(metrics)
        except Exception as exc:
            if classify_failure(exc) == "submitter":
                raise ValueError(str(exc)) from exc
            raise RecoverableException(f"RobotPolicy evaluation unavailable: {exc}") from exc

    def cleanup(self):
        # evaluate_assignment closes the simulator in its finally block.
        pass


MODULE = RobotPolicyValidationModule
