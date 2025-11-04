import numpy as np
from pathlib import Path
import sys


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from validator.modules.rl import (
    RLValidationModule,
    RLConfig,
    RLInputData,
)


def load_demo_data():
    """Load demo numpy data from rl/demo_data directory"""
    demo_dir = Path(__file__).parent / "validator" / "modules" / "rl" / "demo_data"

    X_path = demo_dir / "X_test.npy"
    Info_path = demo_dir / "Info_test.npy"

    if not X_path.exists() or not Info_path.exists():
        print(f"Demo data not found at: {demo_dir}")
        return None, None

    X_test = np.load(X_path)
    Info_test = np.load(Info_path)

    print(f"Loaded X_test with shape: {X_test.shape}")
    print(f"Loaded Info_test with shape: {Info_test.shape}")

    return X_test, Info_test


def test_rl_module():
    """Test RL module using local demo data and model"""
    print("\nTesting RL module with local model...")

    X_test, Info_test = load_demo_data()
    if X_test is None or Info_test is None:
        print("Failed to load demo data")
        return False

    try:
        # Get local model path
        demo_dir = Path(__file__).parent / "validator" / "modules" / "rl" / "demo_data"
        local_model_path = demo_dir / "actor_td3_entropy.onnx"

        if not local_model_path.exists():
            print(f"Model not found at: {local_model_path}")
            return False

        print(f"Using local model: {local_model_path}")

        # Create config (batch_size=1 to match model's expected input dimension)
        config = RLConfig(per_device_eval_batch_size=1, seed=42)

        # Create module
        module = RLValidationModule(config)

        # Mock the data loading to use local files
        def mock_load_data(url):
            if "X_test" in url or "test_X" in url:
                return X_test
            elif "Info_test" in url or "test_Info" in url:
                return Info_test
            else:
                raise ValueError(f"Unknown URL: {url}")

        module._load_data = mock_load_data

        # Mock the model loading to use local file
        import onnxruntime as ort

        def mock_load_model(_repo_id, _filename):
            session = ort.InferenceSession(
                str(local_model_path), providers=["CPUExecutionProvider"]
            )
            print(f"Loaded local ONNX model from {local_model_path}")
            return session

        module._load_model = mock_load_model

        # Create input data (repo_id and filename will be ignored due to mock)
        input_data = RLInputData(
            model_repo_id="test/test_rl_model",
            model_filename="actor_td3_entropy.onnx",
            task_type="reinforcement_learning",
            test_X_url="X_test.npy",
            test_Info_url="Info_test.npy",
            task_id=1,
            required_metrics=["average_reward"],
        )

        # Validate
        print("Running validation...")
        metrics = module.validate(input_data)

        print("Validation completed successfully!")
        print(f"   - Average Reward: {metrics.average_reward}")

        module.cleanup()
        return True

    except Exception as e:
        print(f"Direct validation failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing RL Module")
    print("=" * 50)

    test_passed = test_rl_module()

    print("\n" + "=" * 50)
    if test_passed:
        print("Test PASSED!")
        sys.exit(0)
    else:
        print("Test FAILED")
        sys.exit(1)
