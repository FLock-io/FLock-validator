from typing import Dict, Callable, Any

# Global registry for prompts
_PROMPT_REGISTRY: Dict[int, Callable[[Any], str]] = {}


def register(prompt_id: int):

    def decorator(func: Callable[[Any], str]):
        _PROMPT_REGISTRY[prompt_id] = func
        return func

    return decorator


def get_prompt(
    prompt_id: int,
    data: str,
    reference: str = None,
    tools: str = None,
    assistant_response: str = None,
) -> str:
    """
    Get the registered prompt for a given prompt_id
    """
    if prompt_id not in _PROMPT_REGISTRY:
        raise ValueError(f"No prompt registered for prompt_id {prompt_id}")

    prompt_func = _PROMPT_REGISTRY[prompt_id]

    if prompt_id == 1:
        return prompt_func(data, assistant_response)
    elif prompt_id == 2 and reference:
        return prompt_func(data, reference, assistant_response)
    elif prompt_id == 3 and reference and tools:
        return prompt_func(data, reference, tools, assistant_response)
    else:
        return prompt_func(data)


def list_registered_tasks() -> list[int]:
    return list(_PROMPT_REGISTRY.keys())


@register(prompt_id=1)
def default_evaluation_prompt(context: str, assistant_response: str):
    evaluation_criteria = """The AI assistant has been provided with a conversation history (including prior user queries and assistant replies) as well as system-level instructions. It then generates a final response to the last user query.
    
    Your evaluation should focus only on the final response — do not consider the quality of previous turns — and should be based on the following five criteria. For each criterion, assign a score from 1 to 10, where 1 is the lowest and 10 is the highest:
    - Factuality: Whether the information provided in response is accurate, based on reliable facts and data.
    - User Satisfaction: Whether the responses meets the user's question and needs, and provides a comprehensive and appropriate answer to the question.
    - Logical Coherence: Whether the responses maintains overall consistency and logical coherence between different turns of the conversation, avoiding self-contradiction.
    - Richness: Whether the response includes rich info, depth, context, diversity to meet user needs and provide a comprehensive understanding.
    - Clarity: Whether the response is clear and understandable, and whether it uses concise language and structure so that user can easily understand it.
    - Instruction-following: Whether the response adheres to any specific instructions or guidelines provided by the user or system.
    
    Scoring guidelines:
    - 1-3 points: Poor quality, fails to meet most criteria, contains significant errors or omissions.
    - 4-6 points: Fair quality, meets some criteria but has notable issues,
    - 7-9 points: Good quality, meets most criteria, has minor issues,
    - 10 points: Excellent quality, meets all criteria, no issues.

    # Conversation Context:
    {conversation_context}

    # Assistant's Response:
    {assistant_response}

    Please provide a rationale for your score, your confidence of the score, and specifically addressing the relevance to the user's question in accordance with the criteria above.
    Your confidence of the score should be between 0 and 1, where 1 means you are very sure of the score, and 0 means you are very unsure of the score.

    Your response should be in the following JSON format:
    {{
        "score": <score>,  # A number between 1 and 10
        "confidence": <confidence>,  # A number between 0 and 1
        "reasoning": "<reasoning>"  # Your reasoning for the score
    }}
    """

    return evaluation_criteria.format(
        conversation_context=context, assistant_response=assistant_response
    )


@register(prompt_id=2)
def reference_evaluation_prompt(context: str, reference: str, assistant_response: str):
    evaluation_criteria = """The AI assistant has been provided with a conversation history (including prior user queries and assistant replies) as well as system-level instructions. It then generates a final response to the last user query.
    
    You are also provided with a reference response, which is a high-quality response to the last user query. Your evaluation should focus only on the final response — do not consider the quality of previous turns. When you commence your evaluation, you should follow the following process:
    1. Compare the final AI assistant’s response to the reference answer, pointing out any shortcomings in the AI assistant’s  response and explaining further.
    2. Evaluate the final AI assistant’s response on different dimensions, and after each dimension evaluation, assign a score  from 1 to 10.
    3. Finally, aggregate the assessments from each dimension to give an overall score for the AI assistant’s response,  ranging from 1 to 10.
    4. Overall, the higher  the quality of the model’s response, the higher the score. The dimensions of fact correctness and meeting user needs are  the most important, and these dimensions heavily influence the final composite score.

     — Dimensions to evaluate:
    - Factuality: Whether the information provided in response is accurate, based on reliable facts and data.
    - User Satisfaction: Whether the responses meets the user's question and needs, and provides a comprehensive and appropriate answer to the question.
    - Logical Coherence: Whether the responses maintains overall consistency and logical coherence between different turns of the conversation, avoiding self-contradiction.
    - Richness: Whether the response includes rich info, depth, context, diversity to meet user needs and provide a comprehensive understanding.
    - Clarity: Whether the response is clear and understandable, and whether it uses concise language and structure so that user can easily understand it.
    - Instruction-following: Whether the response adheres to any specific instructions or guidelines provided by the user or system.
    
    Scoring guidelines:
    - 1-3 points: Poor quality, when the model’s response is irrelevant to the conversation, contains significant factual errors, or generates harmful content,  the total score must be 1 to 3 points.
    - 4-6 points: Fair quality, meets some criteria but has notable issues,
    - 7-9 points: Good quality, When the model’s response quality is close to the reference answer in all dimensions and performs wel
    - 10 points: Only when the model’s response quality significantly surpasses the reference answer, adequately addresses the user’s conversation and all requirements, and is close to a perfect score in all dimensions

    # Conversation Context:
    {conversation_context}

    # Reference Response:
    {reference_response}

    # Assistant's Response:
    {assistant_response}

    Please provide a rationale for your score, your confidence of the score, and specifically addressing the relevance to the user's question in accordance with the criteria above.
    Your confidence of the score should be between 0 and 1, where 1 means you are very sure of the score, and 0 means you are very unsure of the score.

    Your response should be in the following JSON format:
    {{
        "score": <score>,  # A number between 1 and 10
        "confidence": <confidence>,  # A number between 0 and 1
        "reasoning": "<reasoning>"  # Your reasoning for the score
    }}
    """

    return evaluation_criteria.format(
        conversation_context=context,
        reference_response=reference,
        assistant_response=assistant_response,
    )


@register(prompt_id=3)
def function_call_ref_eval_prompt(
    context: str, reference: str, Tools: str, assistant_response: str
):

    # This prompt is designed for evaluating function_call responses compared to reference function_call responses.
    # Focused on 1. Format correctness 2. Calling funcitons correctly 3. Parameters correctness 4. Overall quality compared to reference

    evaluation_criteria = """
    The AI assistant has been provided with a user query required tool callings, please evaluate the assistant's function_call response compared to the reference function_call response based on the following criteria:
1. Format Correctness: Whether the function_call response adheres to the expected format, including the presence of necessary fields such as "function_name" and "parameters", and whether these fields are structured correctly.
2. Function Calling Correctness: Whether the function_call response correctly identifies and calls the appropriate function from the list of available tools.
3. Parameters Correctness: Whether the parameters provided in the function_call response are correct and appropriate for the function being called, including correct parameter names and values.

- Scoring guidelines:
- 1-3 points: Poor quality, when the function_call response is in an incorrect format, calls the wrong function, or has incorrect parameters, the total score must be 1 to 3 points.
- 4-6 points: Fair quality, when the function_call response has some issues in format, function calling, or parameters, but still shows some understanding of the requirements.
- 7-9 points: Good quality, when the function_call response is mostly correct, with minor issues in format, function calling, or parameters.
- 10 points: Excellent quality, when the function_call response is correct in format, function calling, and parameters, and demonstrates a high level of understanding of the requirements.

Please note that the function calling may involve multiple steps, and the assistant's response is just one step in the overall function calling process. Therefore, the evaluation should focus on the quality of this specific function_call response based on the conversation context, rather than the overall function calling process.

# Conversation Context:
{conversation_context}

# Available Tools:
{tools}

# Reference Function Call Response:
{reference_response}

# Assistant's Function Call Response:
{assistant_response}

Now, please provide a rationale for your score, your confidence of the score, and specifically addressing the relevance to the user's question in accordance with the criteria above in the following JSON format:
{{
    "score": <score>,  # A number between 1 and 10
    "confidence": <confidence>,  # A number between 0 and 1
    "reasoning": "<reasoning>"  # Your reasoning for the score
}}
    """

    return evaluation_criteria.format(
        conversation_context=context,
        reference_response=reference,
        tools=Tools,
        assistant_response=assistant_response,
    )

template_str= """{% for message in messages %}

{% if message.role == "system" %}
<system>
{{ message.content }}
</system>

{% elif message.role == "user" %}
<user>
{{ message.content }}
</user>

{% elif message.role == "assistant" %}

    {% if message.tool_calls %}
<tool_call>
    {% for tool in message.tool_calls %}
<function={{ tool.function.name }}>
        {% set args = tool.function.arguments %}
        {% if args is string %}
        {% set args = args | from_json %}
        {% endif %}
        {% for key, value in args.items() %}
<parameter={{ key }}>{{ value }}</parameter>
        {% endfor %}
</function>
    {% endfor %}
</tool_call>
    {% else %}
<assistant>
{{ message.content }}
</assistant>
    {% endif %}
{% elif message.role == "tool" %}
<tool_response>
{{ message.content }}
</tool_response>

{% endif %}

{% endfor %}"""