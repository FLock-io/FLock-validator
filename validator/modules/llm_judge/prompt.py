from typing import Dict, Callable, Any

# Global registry for prompts
_PROMPT_REGISTRY: Dict[int, Callable[[Any], str]] = {}


def register(task_id: int):

    def decorator(func: Callable[[Any], str]):
        _PROMPT_REGISTRY[task_id] = func
        return func

    return decorator


def get_prompt(task_id: int, data: str) -> str:
    """
    Get the registered prompt for a given task_id
    """
    if task_id not in _PROMPT_REGISTRY:
        raise ValueError(f"No prompt registered for task_id {task_id}")

    prompt_func = _PROMPT_REGISTRY[task_id]
    return prompt_func(data)


def list_registered_tasks() -> list[int]:
    return list(_PROMPT_REGISTRY.keys())


@register(task_id=1)
def default_evaluation_prompt(context: str):
    evaluation_criteria = """You are a fiar judge, please evaluate the quality of an AI assistant's responses to user queries in a multi-turn conversation.
    Your evaluation should be based on the following criteria:
    - Factuality: Whether the information provided in response is accurate, based on reliable facts and data.
    - User Satisfaction: Whether the responses meets the user's question and needs, and provides a comprehensive and appropriate answer to the question.
    - Logical Coherence: Whether the responses maintains overall consistency and logical coherence between different turns of the conversation, avoiding self-contradiction.
    - Richness: Whether the response includes rich info, depth, context, diversity to meet user needs and provide a comprehensive understanding.
    - Clarity: Whether the response is clear and understandable, and whether it uses concise language and structure so that user can easily understand it.
    
    Scoring guidelines:
    - 1-3 points: Poor quality, fails to meet most criteria, contains significant errors or omissions.
    - 4-6 points: Fair quality, meets some criteria but has notable issues,
    - 7-9 points: Good quality, meets most criteria, has minor issues,
    - 10 points: Excellent quality, meets all criteria, no issues.

    Multi-turn conversation context:
    {context}

    Please provide a rationale for your score, your confidence of the score, and specifically addressing the relevance to the user's question in accordance with the criteria above.
    Your confidence of the score should be between 0 and 1, where 0 means you are very sure of the score, and 1 means you are very unsure of the score.

    Your response should be in the following JSON format:
    {{
        "score": <score>,  # A number between 1 and 10
        "confidence": <confidence>,  # A number between 0 and 1
        "reasoning": "<reasoning>"  # Your reasoning for the score
    }}
    """

    return evaluation_criteria.format(context=context)
