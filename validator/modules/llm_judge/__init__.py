import os
import time
import re
import random
import json
import requests
import httpx
from openai import OpenAI
from typing import List, Optional, Dict, Any
from .prompt import get_prompt
from pydantic import BaseModel
from validator.exceptions import RecoverableException
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .template import template_dict
from loguru import logger


class LLMJudgeException(RecoverableException):

    pass


class LLMJudgeConfig(BaseConfig):
    temperature: float = 0.7
    max_eval_try: int = 3
    max_gen_try: int = 1  # Number of attempts to generate responses for each input


class LLMJudgeMetrics(BaseMetrics):
    score: float  # LLM output score as the main metric
    confidence: Optional[float] = None  # Optional confidence measure
    reasoning: Optional[str] = None  # Optional reasoning from LLM


class LLMJudgeInputData(BaseInputData):
    model_name_or_path: str
    model_template: str
    task_id: int
    test_data_url: str
    evaluation_arg_url: str
    evaluation_criteria: Optional[str] = None


class LLMJudgeValidationModule(BaseValidationModule):

    config_schema = LLMJudgeConfig
    metrics_schema = LLMJudgeMetrics
    input_data_schema = LLMJudgeInputData
    task_type = "llm_evaluation"

    def __init__(self, config: LLMJudgeConfig, **kwargs):
        super().__init__()  # Use kwargs for parent class if needed
        self.config = config
        self.client = None
        self.available_models = []
        self.hf_model = None
        self.hf_tokenizer = None
        # Initialize client and get available models
        self._initialize_client()
        self._fetch_available_models()

    def _initialize_client(self):
        try:
            http_client = httpx.Client(
                base_url=os.getenv("OPENAI_BASE_URL"),
                headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            )

            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                http_client=http_client,
            )

        except Exception as e:
            raise LLMJudgeException(f"Failed to initialize OpenAI client: {e}")

    def _fetch_available_models(self):
        try:
            models_response = self.client.models.list()
            self.available_models = [model.id for model in models_response.data]

        except Exception as e:
            # Fallback to common models if API call fails
            print(
                f"Warning: Failed to fetch models from API ({e}), using fallback models"
            )

    def _load_hf_model(self, model_name_or_path: str):
        try:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            # Add padding token if it doesn't exist
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

        except Exception as e:
            raise LLMJudgeException(
                f"Failed to load HuggingFace model {model_name_or_path}: {e}"
            )

    def _generate_response(
        self,
        user_input: str = None,
        template_name: str = None,
        max_length: int = 512,
        conversation_history: List[Dict[str, str]] = None,
    ) -> str:
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            if template_name not in template_dict:
                logger.warning(f"Template {template_name} not found, using default")
                template_name = "default"

            template = template_dict[template_name]

            conversation_parts = []

            if template.system_format and template.system:
                system_text = template.system_format.format(content=template.system)
                conversation_parts.append(system_text)

            # multi-turn conversation or single user input
            if conversation_history:
                # Multi-turn conversation: format each message according to template
                for msg in conversation_history:
                    if msg["role"] == "user":
                        user_text = template.user_format.format(
                            content=msg["content"],
                            stop_token=self.hf_tokenizer.eos_token,
                        )
                        conversation_parts.append(user_text)
                    elif msg["role"] == "assistant":
                        assistant_text = template.assistant_format.format(
                            content=msg["content"],
                            stop_token=self.hf_tokenizer.eos_token,
                        )
                        conversation_parts.append(assistant_text)
            elif user_input:
                # Single user input
                user_text = template.user_format.format(
                    content=user_input, stop_token=self.hf_tokenizer.eos_token
                )
                conversation_parts.append(user_text)
            else:
                raise LLMJudgeException(
                    "Either user_input or conversation_history must be provided"
                )

            conversation_format = "".join(conversation_parts)

            # Tokenize input
            inputs = self.hf_tokenizer.encode(conversation_format, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = inputs.cuda()

            # Generate response
            with torch.no_grad():
                outputs = self.hf_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id,
                )

            # Decode the generated response
            full_response = self.hf_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )
            # Extract only the assistant's response
            if len(full_response) > len(conversation_format):
                assistant_response = full_response[len(conversation_format) :].strip()

                if template.stop_word and template.stop_word in assistant_response:
                    assistant_response = assistant_response.split(template.stop_word)[
                        0
                    ].strip()

                return assistant_response
            else:
                return ""

        except Exception as e:
            raise LLMJudgeException(f"Failed to generate response: {e}")

    def _select_eval_model(self, eval_args: dict) -> str:
        """
        Select evaluation model based on eval_args configuration
        """
        eval_model_list = eval_args.get("eval_model_list", [])

        if eval_model_list:
            # Check if all models in eval_model_list are available
            available_eval_models = [
                model for model in eval_model_list if model in self.available_models
            ]

            if len(available_eval_models) == len(eval_model_list):
                selected_model = random.choice(eval_model_list)
                print(f"Using eval_model_list: selected {selected_model}")
                return selected_model

        # random selection from available models
        selected_model = random.choice(self.available_models)
        return selected_model

    def _normalize_score(
        self, score: float, min_score: float = 1.0, max_score: float = 10.0
    ) -> float:
        """
        Normalize score to (0, 1) range

        Args:
            score: Original score
            min_score: Minimum possible score (default: 1.0)
            max_score: Maximum possible score (default: 10.0)

        Returns:
            Normalized score in (0, 1) range
        """
        if max_score == min_score:
            return 0.5  # Return middle value if range is zero

        # Normalize to [0, 1] range
        normalized = (score - min_score) / (max_score - min_score)

        # Ensure the result is strictly in (0, 1) range, not [0, 1]
        # Add small epsilon to avoid exact 0 or 1
        epsilon = 1e-8
        normalized = max(epsilon, min(1.0 - epsilon, normalized))

        return normalized

    def _call_gpt(
        self, messages: List[Dict[str, str]], eval_args: dict
    ) -> tuple[str, str]:
        """
        Call GPT API with model and temperature from eval_args

        Args:
            messages: Chat messages
            eval_args: Evaluation arguments containing model and temperature config

        Returns:
            Tuple of (API response content, selected model name)
        """
        selected_model = self._select_eval_model(eval_args)
        temperature = eval_args.get("temperature", self.config.temperature)

        params = {
            "model": selected_model,
            "messages": messages,
            "temperature": temperature,
            "seed": random.randint(0, 10000),
        }

        try:
            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content, selected_model
        except Exception as e:
            raise LLMJudgeException(f"API call failed with model {selected_model}: {e}")

    def _load_data_and_args(
        self, data_url: str, evaluation_arg_url: str
    ) -> tuple[str, dict]:
        # Load test data
        response = requests.get(data_url)
        response.raise_for_status()
        test_data = response.text

        # Load evaluation arguments
        eval_args = {}
        if evaluation_arg_url:
            try:
                eval_response = requests.get(evaluation_arg_url)
                eval_response.raise_for_status()
                eval_args = json.loads(eval_response.text)
                print(f"Loaded evaluation arguments: {eval_args}")
            except Exception as e:
                print(
                    f"Warning: Failed to load evaluation arguments from {evaluation_arg_url}: {e}"
                )

        return test_data, eval_args

    def _load_jsonl_conversations(
        self,
        model_name_or_path: str,
        model_template: str,
        data_url: str,
        evaluation_arg_url: str,
    ) -> List[Dict[str, Any]]:
        """
        Load test data and generate conversations using HuggingFace model

        Args:
            model_name_or_path: HuggingFace model to use for generation
            model_template: Template name for conversation formatting
            data_url: URL to JSONL file containing user inputs
            evaluation_arg_url: URL to evaluation arguments/criteria

        Returns:
            List of generated conversations
        """
        # Load the HuggingFace model
        if self.hf_model is None:
            self._load_hf_model(model_name_or_path)

        # Load data and arguments
        test_data, eval_args = self._load_data_and_args(data_url, evaluation_arg_url)

        # Extract parameters from eval_args
        max_gen_try = eval_args.get("max_gen_try", self.config.max_gen_try)

        generated_conversations = []
        for line_num, line in enumerate(test_data.strip().split("\n"), 1):
            if line.strip():
                try:
                    json_data = json.loads(line)

                    # Extract conversation history for multi-turn support
                    conversation_history = None
                    user_input = None

                    if "conversations" in json_data:
                        conversations = json_data["conversations"]
                        if isinstance(conversations, list) and conversations:
                            # Filter valid messages
                            valid_conversations = []
                            for msg in conversations:
                                role = msg.get("role", "")
                                content = msg.get("content", "").strip()
                                if role in ["user", "assistant"] and content:
                                    valid_conversations.append(
                                        {"role": role, "content": content}
                                    )

                            if valid_conversations:
                                # Check if last message is from user (incomplete conversation)
                                last_msg = valid_conversations[-1]
                                if last_msg["role"] == "user":
                                    # Incomplete conversation - use full history for generation
                                    conversation_history = valid_conversations
                                else:
                                    # Complete conversation - extract last user message for new turn
                                    user_messages = [
                                        msg["content"]
                                        for msg in valid_conversations
                                        if msg["role"] == "user"
                                    ]
                                    if user_messages:
                                        user_input = user_messages[
                                            -1
                                        ]  # Use last user message

                    if not conversation_history and not user_input:
                        user_input = json_data.get("user", "").strip()

                    if not conversation_history and not user_input:
                        print(
                            f"Warning: No user input found in line {line_num}, skipping"
                        )
                        continue

                    # Generate assistant response with defined max_gen_try
                    all_responses = []

                    # Generate multiple responses
                    for gen_try in range(max_gen_try):
                        if conversation_history:
                            assistant_response = self._generate_response(
                                template_name=model_template,
                                conversation_history=conversation_history,
                            )
                            # Use full conversation history + new response
                            final_conversations = conversation_history + [
                                {"role": "assistant", "content": assistant_response}
                            ]
                        else:
                            # Single-turn conversation generation
                            assistant_response = self._generate_response(
                                user_input=user_input, template_name=model_template
                            )
                            # Create simple user-assistant pair
                            final_conversations = [
                                {"role": "user", "content": user_input},
                                {
                                    "role": "assistant",
                                    "content": assistant_response,
                                },
                            ]

                        # Create conversation structure for this generation
                        conversation = {
                            "conversations": final_conversations,
                            "generation_index": gen_try,
                            "total_generations": max_gen_try,
                        }
                        all_responses.append(conversation)

                        print(
                            f"Generated conversation {line_num}/{len(test_data.strip().split(chr(10)))}, generation {gen_try + 1}/{max_gen_try}"
                        )

                    # Add all generated responses for this input
                    generated_conversations.extend(all_responses)

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num}, skipping")
                    continue

        if not generated_conversations:
            raise LLMJudgeException("No valid conversations were generated")

        print(
            f"Successfully generated {len(generated_conversations)} conversations using model {model_name_or_path}"
        )
        return generated_conversations

    def _format_single_conversation(self, json_data: Dict[str, Any]) -> str:
        conversation = json_data.get("conversations", json_data)

        # Convert conversation to formatted string
        if isinstance(conversation, list):
            formatted_conversation = ""
            for i, message in enumerate(conversation):
                if isinstance(message, dict):
                    role = message.get("role", message.get("from", "unknown"))
                    content = message.get("content", message.get("value", str(message)))
                    formatted_conversation += f"**{role.capitalize()}:** {content}\n\n"
                else:
                    formatted_conversation += f"**Message {i+1}:** {str(message)}\n\n"
            return formatted_conversation.strip()
        elif isinstance(conversation, str):
            return conversation
        else:
            return str(conversation)

    def _construct_evaluation_prompt(
        self, conversation_context: str, task_id: int
    ) -> List[Dict[str, str]]:
        """Construct evaluation prompt for a single conversation"""
        try:
            user_message = get_prompt(task_id, conversation_context)
        except ValueError as e:
            user_message = get_prompt(1, conversation_context)
            print(f"Warning: {e}. Using default prompt.")

        system_prompt = """You are a fair judge, please output the score, confidence, and reasoning for the given conversation."""
        return [
            {"role": "user", "content": user_message},
            {"role": "system", "content": system_prompt},
        ]

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        result = {"score": 0.0, "confidence": 0, "reasoning": None}

        try:
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_json = json.loads(json_str)

                if "score" in parsed_json:
                    result["score"] = float(parsed_json["score"])
                if "confidence" in parsed_json:
                    result["confidence"] = float(parsed_json["confidence"])
                if "reasoning" in parsed_json:
                    result["reasoning"] = str(parsed_json["reasoning"])

                return result
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Failed to parse JSON response: {e}")
            return result

    def validate(self, data: LLMJudgeInputData, **kwargs) -> LLMJudgeMetrics:
        """
        Validate using LLM as a judge with multiple tries and random model selection

        Args:
            data: Input data containing evaluation prompt and context
            **kwargs: Additional arguments (currently unused)

        Returns:
            LLMJudgeMetrics: Metrics containing the averaged LLM evaluation score across all conversations
        """
        _ = kwargs

        all_conversations = self._load_jsonl_conversations(
            data.model_name_or_path,
            data.model_template,
            data.test_data_url,
            data.evaluation_arg_url,
        )

        # Load evaluation arguments to get max_eval_try
        _, eval_args = self._load_data_and_args(
            data.test_data_url, data.evaluation_arg_url
        )
        max_eval_try = eval_args.get("max_eval_try", self.config.max_eval_try)

        conversation_scores = []
        conversation_confidences = []
        all_conversation_reasoning = []

        # Group conversations by original input (multiple generations of same input)
        grouped_conversations = {}
        for conv in all_conversations:
            # Create a unique identifier for the original input
            # Use conversation content excluding assistant response to group same inputs
            conversations = conv.get("conversations", [])
            if conversations and len(conversations) >= 2:
                # Use user queries to identify the same input
                user_parts = [
                    msg["content"] for msg in conversations if msg.get("role") == "user"
                ]
                input_key = "|".join(user_parts)
            else:
                input_key = f"single_{len(grouped_conversations)}"

            if input_key not in grouped_conversations:
                grouped_conversations[input_key] = []
            grouped_conversations[input_key].append(conv)

        # Process each group of conversations (multiple generations of same input)
        for group_idx, (input_key, conv_group) in enumerate(
            grouped_conversations.items()
        ):

            group_scores = []
            group_confidences = []
            group_reasoning = []

            # Evaluate each generation in this group
            for gen_idx, conversation_data in enumerate(conv_group):
                conversation_context = self._format_single_conversation(
                    conversation_data
                )
                messages = self._construct_evaluation_prompt(
                    conversation_context, data.task_id
                )

                conv_scores = []
                conv_confidences = []
                conv_reasoning = []

                # Multiple evaluation tries for each generation
                for try_num in range(max_eval_try):
                    response, selected_model = self._call_gpt(messages, eval_args)
                    parsed_result = self._parse_llm_response(response)

                    conv_scores.append(parsed_result["score"])
                    if parsed_result["confidence"] is not None:
                        conv_confidences.append(parsed_result["confidence"])
                    if parsed_result["reasoning"]:
                        conv_reasoning.append(
                            f"Group{group_idx + 1}-Gen{gen_idx + 1}-Try{try_num + 1}({selected_model}): {parsed_result['reasoning']}"
                        )

                # Calculate average for this generation
                if conv_scores:
                    gen_avg_score = sum(conv_scores) / len(conv_scores)
                    group_scores.append(gen_avg_score)

                    if conv_confidences:
                        gen_avg_confidence = sum(conv_confidences) / len(
                            conv_confidences
                        )
                        group_confidences.append(gen_avg_confidence)

                    if conv_reasoning:
                        group_reasoning.extend(conv_reasoning)

            # Calculate average across all generations in this group
            if group_scores:
                group_avg_score = sum(group_scores) / len(group_scores)
                conversation_scores.append(group_avg_score)

                if group_confidences:
                    group_avg_confidence = sum(group_confidences) / len(
                        group_confidences
                    )
                    conversation_confidences.append(group_avg_confidence)

                if group_reasoning:
                    all_conversation_reasoning.extend(group_reasoning)

        # Calculate overall averages across all conversations
        raw_avg_score = sum(conversation_scores) / len(conversation_scores)
        overall_avg_confidence = (
            sum(conversation_confidences) / len(conversation_confidences)
            if conversation_confidences
            else None
        )
        combined_reasoning = (
            "\n\n".join(all_conversation_reasoning)
            if all_conversation_reasoning
            else None
        )

        # Normalize the final score to (0, 1) range
        overall_avg_score = self._normalize_score(raw_avg_score)
        print(f"Overall normalized score (0-1 range): {overall_avg_score:.4f}")

        return LLMJudgeMetrics(
            score=overall_avg_score,
            confidence=overall_avg_confidence,
            reasoning=combined_reasoning,
        )

    def cleanup(self):
        """Clean up resources"""
        if self.client and hasattr(self.client, "http_client"):
            try:
                self.client.http_client.close()
            except Exception:
                pass
        self.client = None

        # Clean up HuggingFace model resources
        if self.hf_model is not None:
            try:
                if torch.cuda.is_available():
                    self.hf_model.cpu()
                    torch.cuda.empty_cache()
                del self.hf_model
            except Exception:
                pass
            self.hf_model = None

        if self.hf_tokenizer is not None:
            del self.hf_tokenizer
            self.hf_tokenizer = None


MODULE = LLMJudgeValidationModule
