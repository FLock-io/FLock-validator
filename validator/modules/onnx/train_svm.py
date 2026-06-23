import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


def load_and_preprocess_data(csv_path):
    """Load and preprocess the training data"""
    print(f"Loading data from {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())

    # Convert Date to datetime and extract features
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear

    # Feature engineering: create lag features
    df = df.sort_values(["store", "product", "Date"])

    # Create lag features for number_sold
    for lag in [1, 2, 3, 7]:
        df[f"number_sold_lag_{lag}"] = df.groupby(["store", "product"])[
            "number_sold"
        ].shift(lag)

    # Create rolling average features
    for window in [3, 7, 14]:
        df[f"number_sold_rolling_{window}"] = (
            df.groupby(["store", "product"])["number_sold"]
            .rolling(window=window)
            .mean()
            .values
        )

    # Drop rows with NaN values (due to lag features)
    df = df.dropna()

    print(f"Data shape after feature engineering: {df.shape}")

    # Prepare features and target
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
    ]

    X = df[feature_columns]
    y = df["number_sold"]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, feature_columns


def train_svm_model(X, y, feature_columns):
    """Train SVM model with preprocessing"""
    print("Training SVM model...")

    if X.shape[0] > 100000:
        subset_size = 100000
        indices = np.random.choice(X.shape[0], subset_size, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
        print(f"Using {subset_size} samples for training")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train SVM model
    svm_model = SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1, max_iter=10000)
    svm_model.fit(X_scaled, y)

    print(f"Training completed on {X.shape[0]} samples")
    return svm_model, scaler, feature_columns


def save_model(model, scaler, feature_columns, save_dir):
    """Save the trained model as ONNX format"""
    os.makedirs(save_dir, exist_ok=True)

    # Convert to ONNX
    try:
        print("Converting model to ONNX format...")

        # Create a pipeline that includes both scaler and model
        pipeline = Pipeline([("scaler", scaler), ("regressor", model)])

        # Define input type for ONNX conversion
        num_features = len(feature_columns)
        initial_type = [("float_input", FloatTensorType([None, num_features]))]

        # Convert to ONNX
        onnx_model = to_onnx(pipeline, initial_types=initial_type)

        # Save ONNX model
        onnx_path = os.path.join(save_dir, "model.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"ONNX model saved to: {onnx_path}")

    except Exception as e:
        print(f"Error: Could not convert to ONNX format: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main training function"""
    print("=== SVM Training Script ===")

    # Set paths
    current_dir = Path(__file__).parent
    csv_path = current_dir / "demo_data" / "train.csv"
    save_dir = current_dir / "trained_svm"

    if not csv_path.exists():
        print(f"Error: Training data not found at {csv_path}")
        return

    try:
        # Load and preprocess data
        X, y, feature_columns = load_and_preprocess_data(csv_path)

        # Train model
        model, scaler, feature_columns = train_svm_model(X, y, feature_columns)

        # Save model
        save_model(model, scaler, feature_columns, save_dir)

        print("\n=== Training completed successfully! ===")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
