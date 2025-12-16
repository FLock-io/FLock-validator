import numpy as np
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)
from validator.exceptions import InvalidModelParametersException
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import onnx
from validator.modules.rl.env import EnvLite
from io import BytesIO
import requests

LOWEST_POSSIBLE_REWARD = -999

class RLConfig(BaseConfig):
    """Configuration for RL validation module"""

    per_device_eval_batch_size: int
    seed: int


class RLMetrics(BaseMetrics):
    """Metrics for RL model validation"""

    average_reward: float


class RLInputData(BaseInputData):
    """Input data for RL validation"""

    hg_repo_id: str
    model_filename: str
    revision: str
    validation_set_url: str
    max_params: int


class RLValidationModule(BaseValidationModule):
    """Validation module for ONNX models"""

    config_schema = RLConfig
    metrics_schema = RLMetrics
    input_data_schema = RLInputData
    task_type = "reinforcement_learning"

    def __init__(self, config: RLConfig, **kwargs):
        self.batch_size = config.per_device_eval_batch_size
        self.seed = config.seed

    def _load_model(self, repo_id: str, filename: str = "model.onnx", revision: str = "main", max_params: int = None):
        """Download and load ONNX model from HuggingFace Hub"""
        model_path = hf_hub_download(repo_id, filename, revision=revision)

        # Try to download external data file if it exists
        # ONNX models with large weights may store data in a separate .onnx.data file
        data_filename = f"{filename}.data"
        try:
            data_path = hf_hub_download(repo_id, data_filename, revision=revision)
            print(f"Downloaded external data file: {data_path}")
        except Exception as e:
            # External data file may not exist, which is fine for models with embedded weights
            print(f"External data file not found (this is OK if model has embedded weights): {e}")

        # Check parameter count
        # onnx.load() will automatically look for .onnx.data in the same directory
        onnx_model = onnx.load(model_path)
        total_params = 0
        for tensor in onnx_model.graph.initializer:
            params = 1
            for dim in tensor.dims:
                params *= dim
            total_params += params

        if max_params and total_params > max_params:
            raise InvalidModelParametersException(f"Model parameters {total_params} exceed limit {max_params}")

        print(f"Model parameters: {total_params}")

        # ort.InferenceSession will also look for .onnx.data in the same directory
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"Loaded ONNX model from {model_path}")
        return session

    def _load_data(self, data_url: str) -> np.ndarray:
        "download and load test data"
        response = requests.get(data_url, timeout=10)
        response.raise_for_status()
        data = np.load(BytesIO(response.content))
        return data

    def validate(self, data: RLInputData, **kwargs) -> RLMetrics:
        """Validate the RL model and compute rewards"""
        # Load model
        try:
            model = self._load_model(data.hg_repo_id, data.model_filename, data.revision, max_params=data.max_params)
        except InvalidModelParametersException as e:
            # lowest possible reward for invalid model parameters
            print(f"Invalid model parameters: {e}")
            return RLMetrics(average_reward=LOWEST_POSSIBLE_REWARD)

        # Download and load test data (.npz file containing X_test and Info_test)
        print(f"Downloading test data from {data.validation_set_url}")
        response = requests.get(data.validation_set_url, timeout=10)
        response.raise_for_status()

        # Load the .npz file and extract X_test and Info_test
        with np.load(BytesIO(response.content)) as test_data:
            test_X = test_data['X']
            test_Info = test_data['Info']

        print(f"Loaded test data: X_test {test_X.shape}, Info_test {test_Info.shape}")

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
