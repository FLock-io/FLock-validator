import json
from typing import List

from loguru import logger
from torch.utils.data import Dataset

from .tool_utils import tool_formater, function_formatter


class UnifiedSFTDataset(Dataset):
    """Dataset that converts conversation JSONL records into language model tokens.

    Each line in the provided JSONL file must follow the same schema that the
    original LoRA validator expects (see `data/dummy_data.jsonl` for an example).
    The template appropriate for the `base_model` argument is fetched from
    `template_dict` and applied to produce text, which is then tokenised.
    """

    def __init__(
        self,
        file: str,
        tokenizer,
        max_seq_length: int,
        template,
    ) -> None:
        self.tokenizer = tokenizer
        self.template_name = template.template_name
        self.system_format = template.system_format
        self.user_format = template.user_format
        self.assistant_format = template.assistant_format
        self.tool_format = template.tool_format
        self.function_format = template.function_format
        self.observation_format = template.observation_format
        self.system = template.system

        self.max_seq_length = max_seq_length
        logger.info(f"Loading evaluation data from {file}")
        with open(file, "r", encoding="utf-8") as f:
            self.data_list: List[str] = f.readlines()
        logger.info(f'Using template "{self.template_name}" for evaluation')
        logger.info(f"Loaded {len(self.data_list)} examples")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = json.loads(self.data_list[index])

        input_ids: List[int] = []
        target_mask: List[int] = []

        # System prompt
        if self.system_format is not None:
            system = data.get("system", self.system)
            if system:
                system_text = self.system_format.format(content=system)
                tokens = self.tokenizer.encode(system_text, add_special_tokens=False)
                input_ids.extend(tokens)
                target_mask.extend([0] * len(tokens))

        # Tool definitions
        if tools_json := data.get("tools"):
            tools = json.loads(tools_json)
            tool_prompt = tool_formater(tools)
            tool_text = self.tool_format.format(content=tool_prompt)
            tokens = self.tokenizer.encode(tool_text, add_special_tokens=False)
            input_ids.extend(tokens)
            target_mask.extend([0] * len(tokens))

        # Conversation turns
        conversations = data["conversations"]
        input_buffer = ""
        for conv in conversations:
            role = conv["role"]
            content = conv["content"].strip()

            if role != "assistant":
                if role == "user":
                    human = self.user_format.format(
                        content=content, stop_token=self.tokenizer.eos_token
                    )
                    input_buffer += human
                elif role == "function_call":
                    tool_calls = function_formatter(json.loads(content))
                    function = self.function_format.format(content=tool_calls)
                    input_buffer += function
                elif role == "observation":
                    observation = self.observation_format.format(content=content)
                    input_buffer += observation
            else:
                assistant = self.assistant_format.format(
                    content=content, stop_token=self.tokenizer.eos_token
                )

                input_tokens = self.tokenizer.encode(
                    input_buffer, add_special_tokens=False
                )
                output_tokens = self.tokenizer.encode(
                    assistant, add_special_tokens=False
                )

                input_ids += input_tokens + output_tokens
                target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)
                input_buffer = ""

        assert len(input_ids) == len(target_mask)

        # truncate
        input_ids = input_ids[: self.max_seq_length]
        target_mask = target_mask[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        } 