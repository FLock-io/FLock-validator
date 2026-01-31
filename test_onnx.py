import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from validator.validation_runner import ValidationRunner
from validator.modules.onnx import (
    ONNXValidationModule,
    ONNXConfig,
    ONNXInputData,
    ONNXMetrics,
)


def load_and_preprocess_demo_data():
    """Load demo CSV data and apply feature engineering"""
    from pathlib import Path

    demo_path = (
        Path(__file__).parent
        / "validator"
        / "modules"
        / "onnx"
        / "demo_data"
        / "test.csv"
    )

    if not demo_path.exists():
        print(f"Demo data not found at: {demo_path}")
        return None

    # Read the demo CSV file
    df = pd.read_csv(demo_path)
    print(f"Loaded demo data with shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear

    df = df.sort_values(["store", "product", "Date"])

    for lag in [1, 2, 3, 7]:
        df[f"number_sold_lag_{lag}"] = df.groupby(["store", "product"])[
            "number_sold"
        ].shift(lag)

    for window in [3, 7, 14]:
        df[f"number_sold_rolling_{window}"] = (
            df.groupby(["store", "product"])["number_sold"]
            .rolling(window=window)
            .mean()
            .values
        )

    df = df.dropna()

    # Select only numerical feature columns
    feature_columns = [
        "store",
        "product",
        "year",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "number_sold_lag_1",
        "number_sold_lag_2",
        "number_sold_lag_3",
        "number_sold_lag_7",
        "number_sold_rolling_3",
        "number_sold_rolling_7",
        "number_sold_rolling_14",
        "number_sold",  # Keep target column
    ]

    df_final = df[feature_columns]

    print(f"After feature engineering: {df_final.shape}")
    print(f"Final columns: {df_final.columns.tolist()}")

    processed_csv = df_final.to_csv(index=False)
    return processed_csv


@patch("validator.validation_runner.FedLedger")
@patch("requests.get")
def test_onnx_validation_works(mock_requests, mock_fedledger):
    """Test that ONNX validation can complete successfully using real HuggingFace model"""

    test_csv = load_and_preprocess_demo_data()
    if test_csv is None:
        print("Failed to load demo data")
        return False

    # Mock API
    mock_api = MagicMock()
    mock_api.list_tasks.return_value = [
        {"id": 1, "task_type": "onnx", "title": "Test", "data": {}}
    ]
    mock_api.mark_assignment_as_failed = MagicMock()
    mock_fedledger.return_value = mock_api

    # Mock HTTP requests for CSV data (use real HuggingFace download for model)
    def mock_get_side_effect(url):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.text = test_csv  # CSV contains both features and target
        return response

    mock_requests.side_effect = mock_get_side_effect

    runner = ValidationRunner(
        module="onnx",
        task_ids=[1],
        flock_api_key="test_key",
        hf_token="test_token",
        test_mode=True,
    )

    input_data = ONNXInputData(
        model_repo_id="Fan9494/test_onnx",
        model_filename="model.onnx",
        revision="main",
        test_data_url="https://example.com/test.csv",
        target_column="number_sold",
        task_type="forecasting",
        task_id=1,
        required_metrics=[
            "mae",
            "rmse",
            "mape",
            "smape",
            "r2_score",
            "directional_accuracy",
        ],
    )

    # Perform validation
    print("Running ONNX validation...")
    metrics = runner.perform_validation("assignment_123", 1, input_data)

    print(f"Validation result: {metrics}")

    if metrics is None:
        print("Validation returned None - something went wrong")
        print("Checking mocks:")
        print(f"  - HTTP requests called: {mock_requests.call_count}")
        return False
    else:
        print("Validation completed successfully!")
        print(f"   - Type: {type(metrics)}")
        if hasattr(metrics, "mae"):
            print(f"   - MAE: {metrics.mae}")
        if hasattr(metrics, "rmse"):
            print(f"   - RMSE: {metrics.rmse}")
        if hasattr(metrics, "mape"):
            print(f"   - MAPE: {metrics.mape}")
        if hasattr(metrics, "smape"):
            print(f"   - SMAPE: {metrics.smape}")

        return True


if __name__ == "__main__":
    print("Testing ONNX Module")
    print("=" * 50)

    # Run tests
    print()
    test_passed = test_onnx_validation_works()

    if test_passed:
        print("\nAll ONNX tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed")
        sys.exit(1)
