import os
import time
import re
import random
import json
import requests
import httpx
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .template import template_dict
from loguru import logger


class LLMJudgeException(RecoverableException):

    pass


class LLMJudgeConfig(BaseConfig):
    gen_batch_size = 1
    eval_batch_size = 10


class LLMJudgeMetrics(BaseMetrics):
    score: float  # LLM output score as the main metric
    confidence: Optional[float] = None  # Optional confidence measure
    reasoning: Optional[str] = None  # Optional reasoning from LLM


class LLMJudgeInputData(BaseInputData):
    model_name_or_path: str
    task_id: int
    test_data_url: str
    evaluation_arg_url: str
    base_model: Optional[str] = None  # For LoRA-adapted models
    evaluation_criteria: Optional[str] = None


class LLMJudgeValidationModule(BaseValidationModule):

    config_schema = LLMJudgeConfig
    metrics_schema = LLMJudgeMetrics
    input_data_schema = LLMJudgeInputData
    task_type = "llm_evaluation"

    def __init__(self, config: LLMJudgeConfig, **kwargs):
        super().__init__(config, **kwargs)
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
            raise LLMJudgeException(f"Failed to initialize OpenAI client: {e}") from e

    def _fetch_available_models(self):
        try:
            models_response = self.client.models.list()
            self.available_models = [model.id for model in models_response.data]

        except Exception as e:
            # Fallback to common models if API call fails
            print(
                f"Warning: Failed to fetch models from API ({e}), using fallback models"
            )
            self.available_models = ["gpt-4o"]

    def _load_hf_model(self, model_name_or_path: str, base_model: str = ""):
        try:
            if base_model:
                # Load LoRA-adapted model
                self.hf_tokenizer = AutoTokenizer.from_pretrained(base_model)
                base_hf_model = AutoModelForCausalLM.from_pretrained(
                    base_model,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
                self.hf_model = PeftModel.from_pretrained(
                    base_hf_model,
                    model_name_or_path,
                    torch_dtype=(
                        torch.float16 if torch.cuda.is_available() else torch.float32
                    ),
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True,
                )
            else:
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
            ) from e

    def _generate_response(
        self,
        user_input: list = list[
            list[dict[str, str]]
        ],  # list of conversations, each conversation is a list of messages
        max_length: int = 7000,
        batch_size: int = 1,
        eval_args: dict = None,
    ) -> str:
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            results = []
            for i in range(0, len(user_input), batch_size):
                batch_conversations = user_input[i : i + batch_size]
                batch_conversation_templates = [
                    self.hf_tokenizer.apply_chat_template(
                        conversation, tokenize=False, add_generation_token=True
                    )
                    for conversation in batch_conversations
                ]
                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    padding_side="left",
                ).to(self.hf_model.device)
                outputs = self.hf_model.generate(
                    **model_inputs,
                    max_new_tokens=max_length,
                    temperature=(
                        eval_args.get("gen_temperature", 0.7) if eval_args else 0.7
                    ),
                    do_sample=True,
                    pad_token_id=self.hf_tokenizer.eos_token_id,
                    eos_token_id=self.hf_tokenizer.eos_token_id,
                )
                for output in outputs:
                    generated_ids = output[model_inputs["input_ids"].shape[1] :]
                    assistant_response = self.hf_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()
                    results.append(assistant_response)

            return results

        except Exception as e:
            raise LLMJudgeException(f"Failed to generate response: {e}") from e

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

        # epsilon = 1e-8
        # normalized = max(epsilon, min(1.0 - epsilon, normalized))

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
        # Check if a specific model is requested
        if "selected_model" in eval_args:
            selected_model = eval_args["selected_model"]
        else:
            selected_model = self._select_eval_model(eval_args)
        temperature = eval_args.get("temperature", 0.3)  # Default eval temperature

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
        data_url: str,
        evaluation_arg_url: str,
        base_model: Optional[str] = None,
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
            self._load_hf_model(model_name_or_path, base_model or "")

        # Load data and arguments
        test_data, eval_args = self._load_data_and_args(data_url, evaluation_arg_url)

        # Extract parameters from eval_args
        max_gen_try = eval_args.get("max_gen_try", 1)  # Default max generation tries

        # Parse all input conversations first
        input_conversations = []
        for line_num, line in enumerate(test_data.strip().split("\n"), 1):
            if line.strip():
                try:
                    json_data = json.loads(line)

                    # Extract system information if available
                    system_text = json_data.get("system", None)

                    # Extract conversation history for multi-turn support
                    conversation_to_process = []

                    if "conversations" in json_data:
                        conversations = json_data["conversations"]
                        if isinstance(conversations, list) and conversations:
                            # Filter valid messages
                            for msg in conversations:
                                role = msg.get("role", "")
                                content = msg.get("content", "").strip()
                                if role in ["user", "assistant"] and content:
                                    conversation_to_process.append(
                                        {"role": role, "content": content}
                                    )

                            # Remove last assistant message if it exists
                            if (
                                conversation_to_process
                                and conversation_to_process[-1]["role"] == "assistant"
                            ):
                                conversation_to_process = conversation_to_process[:-1]

                    # If no conversations found, try to extract from "user" field
                    if not conversation_to_process:
                        user_input = json_data.get("user", "").strip()
                        if user_input:
                            conversation_to_process = [
                                {"role": "user", "content": user_input}
                            ]

                    if not conversation_to_process:
                        print(
                            f"Warning: No user input found in line {line_num}, skipping"
                        )
                        continue

                    # Add system message if present
                    if system_text:
                        conversation_to_process.insert(
                            0, {"role": "system", "content": system_text}
                        )

                    input_conversations.append(
                        {
                            "conversation": conversation_to_process,
                            "line_num": line_num,
                        }
                    )

                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {line_num}, skipping")
                    continue

        if not input_conversations:
            raise LLMJudgeException("No valid conversations were found")

        generated_conversations = []

        # Generate responses for each input conversation
        for gen_try in range(max_gen_try):
            # Prepare batch of conversations for generation
            batch_conversations = [item["conversation"] for item in input_conversations]

            # Generate responses using batch processing
            batch_size = eval_args.get("batch_size", self.config.gen_batch_size)
            assistant_responses = self._generate_response(
                user_input=batch_conversations,
                batch_size=batch_size,
                eval_args=eval_args,
            )

            # Create conversation structures for this generation
            for input_item, assistant_response in zip(
                input_conversations, assistant_responses
            ):
                # Create final conversation with assistant response
                final_conversations = input_item["conversation"] + [
                    {"role": "assistant", "content": assistant_response}
                ]

                # Create conversation structure for this generation
                conversation = {
                    "conversations": final_conversations,
                    "generation_index": gen_try,
                    "total_generations": max_gen_try,
                }
                generated_conversations.append(conversation)

                print(
                    f"Generated conversation {input_item['line_num']}/{len(input_conversations)}, generation {gen_try + 1}/{max_gen_try}"
                )

        if not generated_conversations:
            raise LLMJudgeException("No valid conversations were generated")

        print(
            f"Successfully generated {len(generated_conversations)} conversations using model {model_name_or_path}"
        )
        return generated_conversations

    def _format_single_conversation(self, conversation_data: Dict[str, Any]) -> str:
        """Format a single conversation for evaluation"""
        conversations = conversation_data.get("conversations", [])

        if not conversations:
            return "No conversation found"

        # Format the conversation
        formatted_parts = []

        for msg in conversations:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"User: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        return "\n\n".join(formatted_parts)

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
        result = {"score": 5.0, "confidence": 0, "reasoning": None}

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

    def _evaluate_single_conversation(
        self,
        conversation_data: Dict[str, Any],
        task_id: int,
        eval_args: dict,
        max_eval_try: int,
        group_idx: int,
        gen_idx: int,
    ) -> Dict[str, Any]:
        """
        Evaluate a single conversation and return scores, confidences, and reasoning
        Uses all available models for evaluation, each model runs max_eval_try times
        """
        conversation_context = self._format_single_conversation(conversation_data)
        messages = self._construct_evaluation_prompt(conversation_context, task_id)

        conv_scores = []
        conv_confidences = []
        conv_reasoning = []

        # Get available models for evaluation
        eval_model_list = eval_args.get("eval_model_list", [])
        available_eval_models = [
            model for model in eval_model_list if model in self.available_models
        ]
        
        # If no models specified or available, use all available models
        if not available_eval_models:
            available_eval_models = self.available_models
        
        # Evaluate with each model for max_eval_try times
        for model_idx, model_name in enumerate(available_eval_models):
            for try_num in range(max_eval_try):
                # Create modified eval_args to specify the exact model to use
                model_eval_args = eval_args.copy()
                model_eval_args["selected_model"] = model_name
                
                response, selected_model = self._call_gpt(messages, model_eval_args)
                parsed_result = self._parse_llm_response(response)

                conv_scores.append(parsed_result["score"])
                if parsed_result["confidence"] is not None:
                    conv_confidences.append(parsed_result["confidence"])
                if parsed_result["reasoning"]:
                    conv_reasoning.append(
                        f"Group{group_idx + 1}-Gen{gen_idx + 1}-Model{model_idx + 1}({selected_model})-Try{try_num + 1}: {parsed_result['reasoning']}"
                    )

        return {
            "scores": conv_scores,
            "confidences": conv_confidences,
            "reasoning": conv_reasoning,
        }

    def validate(self, data: LLMJudgeInputData, **kwargs) -> LLMJudgeMetrics:
        """
        Validate using LLM as a judge with multiple tries and random model selection
        Uses two-stage approach: first generate all responses, then parallel evaluation

        Args:
            data: Input data containing evaluation prompt and context
            **kwargs: Additional arguments

        Returns:
            LLMJudgeMetrics: Metrics containing the averaged LLM evaluation score across all conversations
        """
        _ = kwargs

        # Stage 1: Generate all responses
        print("Stage 1: Generating all responses...")
        all_conversations = self._load_jsonl_conversations(
            data.model_name_or_path,
            data.test_data_url,
            data.evaluation_arg_url,
            data.base_model,
        )

        # Load evaluation arguments
        _, eval_args = self._load_data_and_args(
            data.test_data_url, data.evaluation_arg_url
        )
        max_eval_try = eval_args.get("max_eval_try", 3)  # Default max evaluation tries
        eval_batch_size = self.config.eval_batch_size
        
        # Calculate total evaluation calls
        eval_model_list = eval_args.get("eval_model_list", [])
        available_eval_models = [
            model for model in eval_model_list if model in self.available_models
        ]
        if not available_eval_models:
            available_eval_models = self.available_models
        
        total_eval_calls = len(all_conversations) * len(available_eval_models) * max_eval_try

        print(
            f"Stage 2: Evaluating {len(all_conversations)} conversations with {len(available_eval_models)} models, "
            f"{max_eval_try} tries each = {total_eval_calls} total evaluations using {eval_batch_size} parallel workers..."
        )

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

        # Stage 2: Parallel evaluation of all conversations
        conversation_scores = []
        conversation_confidences = []
        all_conversation_reasoning = []

        # Prepare all evaluation tasks
        evaluation_tasks = []
        for group_idx, (input_key, conv_group) in enumerate(
            grouped_conversations.items()
        ):
            for gen_idx, conversation_data in enumerate(conv_group):
                evaluation_tasks.append(
                    (
                        conversation_data,
                        data.task_id,
                        eval_args,
                        max_eval_try,
                        group_idx,
                        gen_idx,
                    )
                )

        # Execute evaluations in parallel
        with ThreadPoolExecutor(max_workers=eval_batch_size) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._evaluate_single_conversation, *task): task
                for task in evaluation_tasks
            }

            # Collect results
            evaluation_results = []
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    evaluation_results.append(result)
                except Exception as e:
                    print(f"Evaluation task failed: {e}")
                    # Add default result for failed tasks
                    evaluation_results.append(
                        {
                            "scores": [5.0],
                            "confidences": [0.5],
                            "reasoning": ["Evaluation failed"],
                        }
                    )

        # Process results and group them back by original input
        result_index = 0
        for group_idx, (input_key, conv_group) in enumerate(
            grouped_conversations.items()
        ):
            group_scores = []
            group_confidences = []
            group_reasoning = []

            # Process results for this group
            for gen_idx in range(len(conv_group)):
                if result_index < len(evaluation_results):
                    result = evaluation_results[result_index]
                    result_index += 1

                    # Calculate average for this generation
                    if result["scores"]:
                        gen_avg_score = sum(result["scores"]) / len(result["scores"])
                        group_scores.append(gen_avg_score)

                    if result["confidences"]:
                        gen_avg_confidence = sum(result["confidences"]) / len(
                            result["confidences"]
                        )
                        group_confidences.append(gen_avg_confidence)

                    if result["reasoning"]:
                        group_reasoning.extend(result["reasoning"])

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
