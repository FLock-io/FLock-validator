from abc import ABC, abstractmethod
from pydantic import BaseModel


class BaseConfig(BaseModel, frozen=True):
    pass


class BaseInputData(BaseModel, frozen=True):
    pass


class BaseMetrics(BaseModel, frozen=True):
    pass


class BaseValidationModule(ABC):
    config_schema: type[BaseConfig]
    input_data_schema: type[BaseInputData]
    metrics_schema: type[BaseMetrics]
    task_type: str

    @abstractmethod
    def __init__(self, config: BaseConfig, **kwargs):
        """.
        Perform any global, one-time setup needed for this module.
        """
        pass

    @abstractmethod
    def validate(self, data: BaseInputData, **kwargs) -> BaseMetrics:
        """
        Download/prep the repo/revision, run validation, and return metrics parsed into a Pydantic model.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up any resources (e.g., temp files, models in memory).
        """
        pass
