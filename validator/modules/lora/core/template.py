from dataclasses import dataclass
from typing import Dict

# NOTE: These templates are adapted from the original LoRA validator implementation.
# They are required for formatting the conversational data when constructing the
# supervised fine-tuning evaluation dataset.

@dataclass
class Template:
    template_name: str
    system_format: str
    user_format: str
    assistant_format: str
    tool_format: str
    function_format: str
    observation_format: str
    system: str | None
    stop_word: str | None


template_dict: Dict[str, Template] = {}


def register_template(
    template_name: str,
    system_format: str,
    user_format: str,
    assistant_format: str,
    tool_format: str,
    function_format: str,
    observation_format: str,
    system: str | None,
    stop_word: str | None = None,
):
    """Register a new formatting template in `template_dict`."""
    template_dict[template_name] = Template(
        template_name=template_name,
        system_format=system_format,
        user_format=user_format,
        assistant_format=assistant_format,
        tool_format=tool_format,
        function_format=function_format,
        observation_format=observation_format,
        system=system,
        stop_word=stop_word,
    )


# A minimal "default" template that works for most base models.
register_template(
    template_name="default",
    system_format="System: {content}\n\n",
    user_format="User: {content}\nAssistant: ",
    assistant_format="{content} {stop_token}",
    tool_format="{content}",
    function_format="{content}",
    observation_format="Tool\n{content}\n",
    system=None,
    stop_word=None,
)

# Register the rest of the templates used by the original script. They are kept
# verbatim so that behaviour remains identical. If you don't need some of these
# models you can safely remove the corresponding template definitions.

register_template(
    template_name="qwen1.5",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|im_start|>tool\n{content}<im_end>\n<|im_start|>assistant\n",
    system="You are a helpful assistant.",
    stop_word="<|im_end|>",
)

register_template(
    template_name="yi",
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|im_start|>tool\n{content}<im_end>\n<|im_start|>assistant\n",
    system=None,
    stop_word="<|im_end|>",
)

register_template(
    template_name="zephyr",
    system_format="<|system|>\n{content}</s>",
    user_format="<|user|>\n{content}</s>\n<|assistant|>\n",
    assistant_format="{content}</s>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|tool|>\n{content}</s>\n<|assistant|>\n",
    system=None,
    stop_word="</s>",
)

register_template(
    template_name="mistral",
    system_format="<s>",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content}</s>",
    tool_format="{content}",
    function_format="{content}",
    observation_format="{content}",
    system="",
    stop_word="</s>",
)

register_template(
    template_name="mixtral",
    system_format="<s>",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content}</s>",
    tool_format="{content}",
    function_format="{content}",
    observation_format="{content}",
    system="",
    stop_word="</s>",
)

register_template(
    template_name="llama2",
    system_format="<<SYS>>\n{content}\n<</SYS>>\n\n",
    user_format="[INST]{content}[/INST]",
    assistant_format="{content} </s>",
    tool_format="{content}",
    function_format="{content}",
    observation_format="{content}",
    system="You are a helpful, respectful and honest assistant. "
    "Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. "
    "If you don't know the answer to a question, please don't share false information.",
    stop_word="</s>",
)

register_template(
    template_name="gemma",
    system_format="<bos>",
    user_format="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    assistant_format="{content}<eos>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<start_of_turn>tool\n{content}<end_of_turn>\n<start_of_turn>model\n",
    system="",
    stop_word="<eos>",
)

register_template(
    template_name="llama3",
    system_format="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
    user_format="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_format="{content}<|eot_id|>",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|start_header_id|>tool<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    system=None,
    stop_word="<|eot_id|>",
)

register_template(
    template_name="phi3",
    system_format=None,
    user_format="<|user|>\n{content}<|end|>\n<|assistant|>",
    assistant_format="{content}<|end|>\n",
    tool_format="{content}",
    function_format="{content}",
    observation_format="<|tool|>\n{content}<|end|>\n<|assistant|>",
    system=None,
    stop_word="<|end|>",
)

register_template(
    template_name="phi4",
    system_format=None,
    user_format="<|user|>\n{content}<|end|>\n<|assistant|>",
    assistant_format="{content}<|end|>\n",
    tool_format="<|tool|>{content}<|/tool|>",
    function_format="<|tool_call|>{content}<|/tool_call|>",
    observation_format="<|tool|>\n{content}<|end|>\n<|assistant|>",
    system=None,
    stop_word="<|end|>",
) 