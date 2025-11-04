import numpy as np
import pandas as pd
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)
from validator.exceptions import RecoverableException
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import os
from validator.modules.rl.env import EnvLite
from io import BytesIO
import requests


class InvalidRLEnvironmentException(RecoverableException):
    pass


class RLConfig(BaseConfig):
    """Configuration for RL validation module"""

    per_device_eval_batch_size: int
    seed: int


class RLMetrics(BaseMetrics):
    """Metrics for RL model validation"""

    average_reward: float


class RLInputData(BaseInputData):
    """Input data for RL validation"""

    model_repo_id: str  # HuggingFace repository ID for the RL model
    model_filename: str  # Name of the ONNX model file

    task_type: str
    test_X_url: str
    test_Info_url: str


class RLValidationModule(BaseValidationModule):
    """Validation module for ONNX models"""

    config_schema = RLConfig
    metrics_schema = RLMetrics
    input_data_schema = RLInputData
    task_type = "reinforcement_learning"

    def __init__(self, config: RLConfig, **kwargs):
        self.batch_size = config.per_device_eval_batch_size
        self.seed = config.seed

    def _load_model(self, repo_id: str, filename: str = "model.onnx"):
        """Download and load ONNX model from HuggingFace Hub"""
        model_path = hf_hub_download(repo_id, filename)
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"Loaded ONNX model from {model_path}")
        return session

    def _load_data(self, data_url: str) -> np.ndarray:
        "download and load testa data"
        response = requests.get(data_url)
        response.raise_for_status()
        data = np.load(BytesIO(response.content))
        return data

    def validate(self, data: RLInputData, **kwargs) -> RLMetrics:
        """Validate the RL model and compute rewards"""
        # Load model and data
        model = self._load_model(data.model_repo_id, data.model_filename)
        test_X = self._load_data(data.test_X_url)
        test_Info = self._load_data(data.test_Info_url)
        env = EnvLite(test_X, test_Info, batch_size=self.batch_size, seed=self.seed)

        N = env.N  # total samples
        all_rewards = []

        # Run evaluation through all samples
        for start_idx in range(0, N, self.batch_size):
            end_idx = min(start_idx + self.batch_size, N)
            batch_indices = np.arange(start_idx, end_idx)
            env.idx = batch_indices
            env.X_b = env.X_all[batch_indices]
            env.Info_b = env.Info_all[batch_indices]
            env.qty_b = env.qty_all[batch_indices]
            env.duration_b = env.duration_all[batch_indices]
            env.fill_b = env.fill_all[batch_indices, :]
            env.rebate_b = env.rebate_all[batch_indices, :]
            env.punish_b = env.punish_all[batch_indices, :]
            env.vol_b = env.vol_all[batch_indices, :]

            # get model input and output names
            input_name = model.get_inputs()[0].name

            # get model actions and rewards
            # model.run() returns a list, [0] gets first output, [0] gets first batch element
            outputs = model.run(None, {input_name: env.X_b})
            action = outputs[0]
            reward = env.step(action)
            all_rewards.append(reward)

        # Compute average reward
        all_rewards = np.concatenate(all_rewards)
        average_reward = float(np.mean(all_rewards))
        return RLMetrics(average_reward=average_reward)

    def cleanup(self):
        """Cleanup resources if needed"""
        pass


MODULE = RLValidationModule
