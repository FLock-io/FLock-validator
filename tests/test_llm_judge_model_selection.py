import pytest

from validator.exceptions import LLMJudgeException
from validator.modules.llm_judge_model_selection import resolve_eval_models


def test_configured_models_are_authoritative():
    models = resolve_eval_models(
        {"eval_model_list": ["kimi-k2.6-thinking", "qwen3.5", "qwen3.5"]},
        ["gpt-4o"],
    )

    assert models == ["kimi-k2.6-thinking", "qwen3.5"]


def test_configured_models_work_when_discovery_fails():
    models = resolve_eval_models(
        {"eval_model_list": ["kimi-k2.6"]},
        [],
    )

    assert models == ["kimi-k2.6"]


def test_discovered_models_are_used_when_config_is_absent():
    models = resolve_eval_models({}, ["model-a", "model-b", "model-a"])

    assert models == ["model-a", "model-b"]


def test_missing_config_and_discovery_fails_explicitly():
    with pytest.raises(LLMJudgeException, match="eval_model_list"):
        resolve_eval_models({}, [])


@pytest.mark.parametrize(
    "configured_models",
    ["model-a", "", None, ["model-a", ""], ["model-a", None]],
)
def test_invalid_configured_models_fail_explicitly(configured_models):
    with pytest.raises(LLMJudgeException, match="eval_model_list"):
        resolve_eval_models({"eval_model_list": configured_models}, ["gpt-4o"])
