from collections.abc import Iterable, Mapping
from typing import Any

from validator.exceptions import LLMJudgeException


def resolve_eval_models(
    eval_args: Mapping[str, Any] | None,
    available_models: Iterable[str],
) -> list[str]:
    requested_models = eval_args.get("eval_model_list", []) if eval_args else []

    if not isinstance(requested_models, list):
        raise LLMJudgeException("eval_model_list must be a list of model names")

    if requested_models:
        if not all(
            isinstance(model, str) and model.strip() for model in requested_models
        ):
            raise LLMJudgeException(
                "eval_model_list must contain only non-empty model names"
            )
        return list(dict.fromkeys(requested_models))

    discovered_models = [
        model
        for model in available_models
        if isinstance(model, str) and model.strip()
    ]
    discovered_models = list(dict.fromkeys(discovered_models))
    if discovered_models:
        return discovered_models

    raise LLMJudgeException(
        "No evaluation models configured and provider model discovery returned none. "
        "Set eval_args.eval_model_list explicitly."
    )
