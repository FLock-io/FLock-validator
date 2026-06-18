import json
from pathlib import Path
from typing import Type, Any
from .modules.base import BaseConfig

def load_config_for_task(
    task_id: str,
    task_type: str,
    config_model: Type[BaseConfig],
    config_dir: str = "configs"
) -> BaseConfig:
    """
    Loads and merges config for a given task_id and task_type, using the provided Pydantic model.
    Priority: per-task-id > per-task-type > model defaults.
    """
    config_data: dict[str, Any] = {}

    # 1. Load per-task-type config
    type_config_path = Path(config_dir) / f"{task_type}.json"
    if type_config_path.exists():
        with open(type_config_path, "r") as f:
            config_data.update(json.load(f))

    # 2. Load per-task-id config (overrides type config)
    task_config_path = Path(config_dir) / "tasks" / f"{task_id}.json"
    if task_config_path.exists():
        with open(task_config_path, "r") as f:
            config_data.update(json.load(f))

    # 3. Use Pydantic model defaults for missing values
    return config_model(**config_data)


def load_config_from_file(
    task_type: str,
    config_model: Type[BaseConfig],
    config_dir: str = "configs"
) -> BaseConfig:
    """
    Loads config for a given task_type from file (for local testing).
    """
    config_data: dict[str, Any] = {}
    type_config_path = Path(config_dir) / f"{task_type}.json"
    if type_config_path.exists():
        with open(type_config_path, "r") as f:
            config_data.update(json.load(f))
    return config_model(**config_data)