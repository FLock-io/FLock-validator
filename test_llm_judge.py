#!/usr/bin/env python3
import os
import sys
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the validator path to sys.path
current_dir = Path(__file__).parent
validator_path = current_dir / "validator"
sys.path.insert(0, str(validator_path))

from validator.modules.llm_judge import (
    LLMJudgeValidationModule,
    LLMJudgeConfig,
    LLMJudgeInputData,
)

from dotenv import load_dotenv

load_dotenv()


def create_mock_response(content: str, status_code: int = 200):
    mock_response = MagicMock()
    mock_response.text = content
    mock_response.status_code = status_code
    mock_response.raise_for_status.return_value = None
    return mock_response


def load_local_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def mock_requests_get(url: str):
    if "test_data.jsonl" in url:
        test_data_path = (
            Path(__file__).parent
            / "validator/modules/llm_judge/test_data"
            / "final_validation_set.jsonl"
        )
        content = load_local_file(test_data_path)
        return create_mock_response(content)

    elif "eval_args.json" in url:
        eval_args = {
            "eval_model_list": ["qwen3-235b-a22b-instruct-2507"],
            "gen_temperature": 0.1,
            "eval_temperature": 0.5,
            "max_eval_try": 1,
            "max_gen_try": 1,
        }
        content = json.dumps(eval_args, indent=2)
        return create_mock_response(content)

    else:
        return create_mock_response('{"error": "Mock URL not configured"}', 404)


def test_llm_judge():
    required_env_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    for var in required_env_vars:
        if not os.getenv(var):
            print(f"Warning: {var} not set. Set it before running the test.")

    # Enable CUDA debugging if available
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Create configuration with smaller batch sizes for stability
    config = LLMJudgeConfig(gen_batch_size=1, eval_batch_size=1)

    # Create input data with a simpler, more stable model
    input_data = LLMJudgeInputData(
        model_name_or_path="jenniellama/task-10-Qwen-Qwen2.5-7B-Instruct",
        task_id=1,
        test_data_url="https://mock.example.com/test_data.jsonl",
        evaluation_arg_url="https://mock.example.com/eval_args.json",
        model_template="qwen1.5",
        base_model="Qwen/Qwen2.5-7B-Instruct",  # No LoRA adapter for this test
        evaluation_criteria="Test evaluation criteria",
    )

    print("Initializing LLM Judge module...")

    # Mock requests.get to use local files
    with patch(
        "validator.modules.llm_judge.requests.get", side_effect=mock_requests_get
    ):
        judge_module = LLMJudgeValidationModule(config)

        try:
            print("Running validation...")

            # Run validation
            metrics = judge_module.validate(input_data)

            print("\n=== Test Results ===")
            print(f"Final Score (normalized): {metrics.score:.4f}")
            print(
                f"Confidence: {metrics.confidence:.4f}"
                if metrics.confidence
                else "Confidence: None"
            )
            print(
                f"Reasoning length: {len(metrics.reasoning) if metrics.reasoning else 0} characters"
            )

            if metrics.reasoning:
                print("\nReasoning sample:")
                print(
                    metrics.reasoning[:200] + "..."
                    if len(metrics.reasoning) > 200
                    else metrics.reasoning
                )

            print("\n=== Test Completed Successfully ===")

        except Exception as e:
            print(f"Error during validation: {e}")
            raise

        finally:
            # Cleanup
            judge_module.cleanup()


if __name__ == "__main__":
    print("LLM Judge Test Script")
    print("=" * 50)

    # test_data_file = (
    #     Path(__file__).parent
    #     / "validator/modules/llm_judge/test_data"
    #     / "final_validation_set.jsonl"
    # )
    # if not test_data_file.exists():
    #     print(f"Test data file not found: {test_data_file}")
    #     print("Please create the test data file first.")
    #     sys.exit(1)

    # print(f"Using test data file: {test_data_file}")

    try:
        test_llm_judge()
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback

        traceback.print_exc()
