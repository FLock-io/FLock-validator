from validator.exceptions import RecoverableException
from validator.modules.base import BaseValidationModule, BaseConfig, BaseInputData, BaseMetrics

# When raised, the assignment won't be marked as failed automatically and it will be retried after the user
# fixes the problem and restarts the process.
class InvalidConfigValueException(RecoverableException):
    pass


class LoRAConfig(BaseConfig):
    per_device_eval_batch_size: int
    fp16: bool
    output_dir: str
    remove_unused_columns: bool

class LoRAMetrics(BaseMetrics):
    loss: float
    bpc: float
    bppl: float
    nll_token_nats_total: float
    nll_token_bits_total: float

class LoRAInputData(BaseInputData):
    hg_repo_id: str
    revision: str
    base_model: str
    eval_file: str
    context_length: int
    max_params: int
    validation_args_file: str

class LoRAValidationModule(BaseValidationModule):
    config_schema = LoRAConfig
    metrics_schema = LoRAMetrics
    input_data_schema = LoRAInputData
    task_type = "training"

    def __init__(self, config: LoRAConfig, **kwargs):
        # Any global setup
        pass

    def validate(self, data: LoRAInputData, **kwargs) -> LoRAMetrics:
        # Download/prep repo, run validation, compute metrics
        # Example dummy result:
        result = {
            "loss": 1.23,
            "bpc": 0.98,
            "bppl": 1.11,
            "nll_token_nats_total": 2.22,
            "nll_token_bits_total": 3.33,
        }

        if False:
            raise InvalidConfigValueException("Invalid config value, recoverable")
        # Validate and return as Pydantic model
        return LoRAMetrics(**result)

    def cleanup(self):
        # Free resources
        pass

MODULE = LoRAValidationModule