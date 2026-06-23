import numpy as np
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel
from validator.exceptions import RecoverableException
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import os


# When raised, the assignment won't be marked as failed automatically and it will be retried after the user
# fixes the problem and restarts the process.
class InvalidTimeSeriesDataException(RecoverableException):
    pass


class ONNXConfig(BaseConfig):
    """Configuration for ONNX validation module"""

    per_device_eval_batch_size: int
    sequence_length: int
    model_type: str
    evaluation_metrics: List[str]
    output_dir: str


class ONNXMetrics(BaseMetrics):
    """Metrics for time series model validation"""

    mae: Optional[float] = None  # Mean Absolute Error
    rmse: Optional[float] = None  # Root Mean Square Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    smape: Optional[float] = None  # Symmetric Mean Absolute Percentage Error
    mse: Optional[float] = None  # Mean Square Error
    r2_score: Optional[float] = None  # R-squared score
    directional_accuracy: Optional[float] = (
        None  # Percentage of correct directional predictions
    )
    forecasting_skill: Optional[float] = None  # Optional skill score vs baseline


class ONNXInputData(BaseInputData):
    """Input data for ONNX validation"""

    model_repo_id: str  # HuggingFace repository ID for the ONNX model
    model_filename: str  # Name of the ONNX model file
    revision: Optional[str] = None

    test_data_url: str
    target_column: str  # Column name in test_data that contains ground truth values

    task_type: str
    task_id: int

    training_set_url: Optional[str] = None

    required_metrics: Optional[List[str]] = None  # Required metrics for this task


class ONNXValidationModule(BaseValidationModule):
    """Validation module for time series models"""

    config_schema = ONNXConfig
    metrics_schema = ONNXMetrics
    input_data_schema = ONNXInputData
    task_type = "time_series_forecasting"

    def __init__(self, config: ONNXConfig, **kwargs):
        """Initialize the OONX validation module"""
        self.config = config
        self.model = None
        self.data_processor = None

    def _load_model(
        self, model_repo_id: str, filename: str = "model.onnx", revision: str = None
    ):
        """Load ONNX model from HuggingFace repository"""
        try:
            model_path = hf_hub_download(
                repo_id=model_repo_id,
                filename=filename,  # perhaps this should be configurable
                revision=revision,
                cache_dir=self.config.output_dir,
            )

            self.model = ort.InferenceSession(model_path)

            print(f"Successfully loaded ONNX model from {model_repo_id}")
            return model_path

        except Exception as e:
            print(f"Error loading model from {model_repo_id}: {e}")
            raise ValueError(f"Failed to load ONNX model: {e}")

    def _load_data(self, data_url: str, target_column: str = None) -> tuple:
        """Load CSV data from URL, optionally separating features and target"""
        import requests
        from io import StringIO

        try:
            response = requests.get(data_url)
            response.raise_for_status()

            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)

            if df.empty:
                raise ValueError("CSV file is empty")

            if target_column:
                # Separate features and target
                if target_column not in df.columns:
                    raise ValueError(
                        f"Target column '{target_column}' not found in data"
                    )

                target_values = df[target_column].values
                feature_values = df.drop(columns=[target_column]).values
                return feature_values, target_values
            else:
                # Return all data as features
                return df.values, None

        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return None, None

    def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
        """Run ONNX model inference"""
        if self.model is None:
            raise ValueError("Model not loaded. Call _load_model first.")

        try:
            input_name = self.model.get_inputs()[0].name

            input_data = input_data.astype(np.float32)

            if len(input_data.shape) == 1:
                input_data = input_data.reshape(1, -1)

            outputs = self.model.run(None, {input_name: input_data})

            return outputs[0].flatten()

        except Exception as e:
            print(f"Error during inference: {e}")
            raise ValueError(f"ONNX inference failed: {e}")

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, task_data: dict = None
    ) -> dict:
        """Compute time series forecasting metrics based on task data"""

        if task_data:
            task_type = task_data.get("task_type", "")
            required_metrics = task_data.get("required_metrics", [])

            if required_metrics:
                requested_metrics = required_metrics
            else:

                if task_type == "training":
                    requested_metrics = ["mae", "rmse", "r2_score"]
                elif task_type == "forecasting":
                    requested_metrics = ["mae", "rmse", "mape", "smape"]
                elif task_type == "classification":
                    requested_metrics = ["mae", "directional_accuracy"]
                elif task_type == "time_series":
                    requested_metrics = ["mae", "rmse", "mape", "directional_accuracy"]
                else:
                    requested_metrics = self.config.evaluation_metrics
        else:
            requested_metrics = self.config.evaluation_metrics

        metrics_dict = {}

        if "mae" in requested_metrics:
            metrics_dict["mae"] = np.mean(np.abs(y_true - y_pred))

        if "rmse" in requested_metrics:
            metrics_dict["rmse"] = np.sqrt(np.mean((y_true - y_pred) ** 2))

        if "mse" in requested_metrics:
            metrics_dict["mse"] = np.mean((y_true - y_pred) ** 2)

        if "mape" in requested_metrics:
            mask = y_true != 0
            if np.any(mask):
                metrics_dict["mape"] = (
                    np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                )
            else:
                metrics_dict["mape"] = 0.0

        if "smape" in requested_metrics:
            denominator = np.abs(y_true) + np.abs(y_pred)
            mask = denominator != 0
            if np.any(mask):
                metrics_dict["smape"] = (
                    np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])
                    * 100
                )
            else:
                metrics_dict["smape"] = 0.0

        if "r2_score" in requested_metrics:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            metrics_dict["r2_score"] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        if "directional_accuracy" in requested_metrics:
            if len(y_true) > 1:
                true_direction = np.sign(np.diff(y_true))
                pred_direction = np.sign(np.diff(y_pred))
                metrics_dict["directional_accuracy"] = (
                    np.mean(true_direction == pred_direction) * 100
                )
            else:
                metrics_dict["directional_accuracy"] = 0.0

        return metrics_dict

    def validate(
        self, data: ONNXInputData, task_id: str = None, **kwargs
    ) -> ONNXMetrics:
        """
        Validate time series model performance

        Args:
            data: Input data containing model and dataset information
            task_id: Task ID for determining metrics

        Returns:
            ONNXMetrics: Computed metrics for the time series model
        """
        try:
            filename = getattr(data, "model_filename", "model.onnx")
            self._load_model(data.model_repo_id, filename, data.revision)

            # Load test data and extract features and target
            test_features, ground_truth = self._load_data(
                data.test_data_url, data.target_column
            )

            if test_features is None or ground_truth is None:
                raise InvalidTimeSeriesDataException(
                    "Failed to load test data or extract ground truth from target column"
                )

            if len(test_features) != len(ground_truth):
                raise InvalidTimeSeriesDataException(
                    "Test features and ground truth length mismatch"
                )

            predictions = self._run_inference(test_features)

            task_data = {
                "task_type": data.task_type,
                "required_metrics": data.required_metrics,
            }

            metrics_dict = self._compute_metrics(
                ground_truth.flatten(), predictions.flatten(), task_data
            )
            return ONNXMetrics(**metrics_dict)

        except Exception as e:
            if isinstance(e, InvalidTimeSeriesDataException):
                raise e
            else:
                raise ValueError(f"Validation failed: {str(e)}")

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.data_processor is not None:
            del self.data_processor
            self.data_processor = None


MODULE = ONNXValidationModule
