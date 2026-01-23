import os
import re
import random
import json
import httpx
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from huggingface_hub import HfApi
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from validator.modules.llm_judge.prompt import get_prompt
from validator.modules.llm_judge.utils import download_file
from validator.exceptions import LLMJudgeException, InvalidModelParametersException
from validator.modules.llm_judge.template import template_dict
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from validator.modules.base import (
    BaseValidationModule,
    BaseConfig,
    BaseInputData,
    BaseMetrics,
)

api = HfApi()
LOWEST_POSSIBLE_SCORE = -999


class LLMJudgeConfig(BaseConfig):
    gen_batch_size: int = 1
    eval_batch_size: int = 10
    gen_temperature: float = 0.7


class LLMJudgeMetrics(BaseMetrics):
    score: float  # LLM output score as the main metric


class LLMJudgeInputData(BaseInputData):
    """Input data for LLMJudge"""
    hg_repo_id: str
    revision: str
    context_length: int
    max_params: int
    validation_set_url: str
    base_model: str
    eval_args: dict


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
            raise LLMJudgeException(f"OPENAI_API_KEY and OPENAI_BASE_URL are not set in the environment variables: {e}") from e

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

    def _download_lora_config(self, repo_id: str, revision: str) -> bool:
        try:
            api.hf_hub_download(
                repo_id=repo_id,
                filename="adapter_config.json",
                local_dir="judge",
                revision=revision,
            )
        except Exception as e:
            if "adapter_config.json" in str(e):
                print("No adapter_config.json found in the repo, assuming full model")
                return False
            else:
                raise  # Re-raise the exception if it's not related to the missing file
        return True

    def _load_model(self, repo_id: str, revision: str = "main", max_params: int = None):

        is_lora = self._download_lora_config(repo_id, revision=revision)
        model_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_cache=False,
            device_map=None,
        )
        if is_lora:
            api.snapshot_download(
                repo_id=repo_id,
                local_dir="judge",
                revision=revision,
            )
            with open("judge/adapter_config.json", "r") as f:
                adapter_config = json.load(f)
            base_model = adapter_config["base_model_name_or_path"]
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True, use_fast=True
            )
            base_hf_model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
            hf_model = PeftModel.from_pretrained(
                base_hf_model,
                "judge",
                device_map=None,
            )
            self.hf_model = hf_model.merge_and_unload()
        else:
            self.hf_tokenizer = AutoTokenizer.from_pretrained(
                repo_id, trust_remote_code=True, use_fast=True
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                device_map=None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        total = sum(p.numel() for p in self.hf_model.parameters())
        if total > max_params:
            print(
                f"Total model params: {total} exceeds the limit {max_params}, submitting validation result with a large loss"
            )
            raise InvalidModelParametersException(f"Model parameters {total} exceed limit {max_params}")

    def _construct_conversation_template(
            self, conversation: List[Dict[str, str]], base_model: str, max_seq_length: int,
    ) -> str:
        try:
            if base_model not in template_dict:
                print(f"Template {base_model} not found, using default")
                base_model = "default"

            template = template_dict[base_model]

            conversation_parts = []

            # Use provided system_text or fall back to template default
            if template.system_format:
                system_prompt = (
                    conversation["system"] if "system" in conversation else None
                )
                system_content = (
                    system_prompt if system_prompt else "You are a helpful assistant."
                )
                if system_content:
                    formatted_system = template.system_format.format(
                        content=system_content
                    )
                    conversation_parts.append(formatted_system)

                # Multi-turn conversation: format each message according to template
                for msg in conversation["conversations"]:
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

            conversation_format = "".join(conversation_parts)
            conversation_format = conversation_format[:max_seq_length]
        except Exception as e:
            raise LLMJudgeException(
                f"Failed to construct conversation template: {e}"
            ) from e

        return conversation_format

    def _generate_response(
            self,
            context_length: int,
            user_input: list = list[
                list[dict[str, str]]
            ],  # list of conversations, each conversation is a list of messages
            base_model: str = "default",
            max_length: int = 2048,
            batch_size: int = 1,
            eval_args: dict = None,

    ) -> list:
        if self.hf_model is None or self.hf_tokenizer is None:
            raise LLMJudgeException("HuggingFace model not loaded")

        try:
            results = []
            for i in range(0, len(user_input), batch_size):
                batch_conversations = user_input[i: i + batch_size]

                # Apply chat template with fallback
                batch_conversation_templates = []
                for conversation in batch_conversations:
                    template = self._construct_conversation_template(
                        conversation, base_model=base_model, max_seq_length=context_length
                    )

                    batch_conversation_templates.append(template)

                # Simple tokenization
                model_inputs = self.hf_tokenizer(
                    batch_conversation_templates,
                    return_tensors="pt",
                    add_special_tokens=True,
                )

                # Move to device if available
                if (
                        torch.cuda.is_available()
                        and next(self.hf_model.parameters()).is_cuda
                ):
                    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

                with torch.no_grad():
                    outputs = self.hf_model.generate(
                        **model_inputs,
                        max_new_tokens=max_length,
                        temperature=self.config.gen_temperature,
                        do_sample=True,
                        pad_token_id=self.hf_tokenizer.eos_token_id,
                        eos_token_id=self.hf_tokenizer.eos_token_id,
                    )

                # Decode responses
                for j, output in enumerate(outputs):
                    # Get the input length for this specific sequence
                    input_length = model_inputs["input_ids"][j].shape[0]

                    # Extract only the newly generated tokens
                    generated_ids = output[input_length:]
                    assistant_response = self.hf_tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    ).strip()

                    results.append(assistant_response)
                    # print("assistant_response:", assistant_response)
                print(f"Generated batch of size {i + batch_size}/{len(user_input)}")

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

    def _load_jsonl_conversations(
            self,
            base_model: str,
            test_file: str,
            eval_args: dict,
            context_length: int
    ) -> List[Dict[str, Any]]:

        # Extract parameters from eval_args
        max_gen_try = eval_args.get("max_gen_try", 1)  # Default max generation tries

        # Parse all input conversations first
        input_conversations = []
        import json

        # TODO: NEED FIX BIG DATASET
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        for line_num, json_data in enumerate(test_data):

            try:
                # Extract system information if available
                system_text = json_data.get("system", None)
                input_conversations_data = {
                    "system": system_text,
                    "conversations": [],
                }

                # Extract conversation history for multi-turn support
                conversation_to_process = []
                reference_response = None

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

                        # Extract reference response (last assistant message) if it exists
                        reference_response = None
                        if (
                                conversation_to_process
                                and conversation_to_process[-1]["role"] == "assistant"
                        ):
                            reference_response = conversation_to_process[-1]["content"]
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

                input_conversations_data["conversations"] = conversation_to_process

                input_conversations.append(
                    {
                        "conversation": input_conversations_data,
                        "line_num": line_num,
                        "reference": reference_response,
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
            assistant_responses = self._generate_response(
                user_input=batch_conversations,
                base_model=base_model,
                batch_size=self.config.gen_batch_size,
                eval_args=eval_args,
                context_length=context_length,
            )

            # Create conversation structures for this generation
            for input_item, assistant_response in zip(
                    input_conversations, assistant_responses
            ):
                # Create final conversation with assistant response
                final_conversations = (
                        [
                            {
                                "role": "system",
                                "content": input_item["conversation"]["system"],
                            }
                        ]
                        + input_item["conversation"]["conversations"]
                        + [{"role": "assistant", "content": assistant_response}]
                )

                # Create conversation structure for this generation
                conversation = {
                    "conversations": final_conversations,
                    "generation_index": gen_try,
                    "total_generations": max_gen_try,
                    "reference": input_item.get("reference"),
                }
                generated_conversations.append(conversation)

        if not generated_conversations:
            raise LLMJudgeException("No valid conversations were generated")

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
            self, conversation_context: str, prompt_id: int, reference: str = None
    ) -> List[Dict[str, str]]:
        """Construct evaluation prompt for a single conversation"""
        try:
            if reference and prompt_id == 2:
                # Use reference evaluation prompt
                user_message = get_prompt(prompt_id, conversation_context, reference)
            else:
                user_message = get_prompt(prompt_id, conversation_context)
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
            eval_args: dict,
            max_eval_try: int,
            conv_idx: int,
    ) -> Dict[str, Any]:
        """
        Simplified version: evaluate a single conversation and return scores, confidences, and reasoning
        """
        conversation_context = self._format_single_conversation(conversation_data)
        reference = conversation_data.get("reference")
        messages = self._construct_evaluation_prompt(conversation_context, eval_args.get("prompt_id", 0), reference)

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
                        f"Conv{conv_idx}-Model{model_idx + 1}({selected_model})-Try{try_num + 1}: {parsed_result['reasoning']}"
                    )

        return {
            "scores": conv_scores,
            "confidences": conv_confidences,
            "reasoning": conv_reasoning,
        }

    def validate(self, data: LLMJudgeInputData, **kwargs) -> LLMJudgeMetrics:
        eval_file = download_file(data.validation_set_url)

        try:
            self._load_model(data.hg_repo_id, data.revision, data.max_params)
        except InvalidModelParametersException as e:
            # lowest possible reward for invalid model parameters
            print(f"Invalid model parameters: {e}")
            return LLMJudgeMetrics(score=LOWEST_POSSIBLE_SCORE)

        # Stage 1: Generate all responses
        print("Stage 1: Generating all conversations for evaluation...")
        all_conversations = self._load_jsonl_conversations(
            data.base_model,
            eval_file,
            data.eval_args,
            data.context_length
        )

        # Load evaluation arguments

        max_eval_try = data.eval_args.get("max_eval_try", 3)  # Default max evaluation tries
        eval_batch_size = self.config.eval_batch_size

        # Calculate total evaluation calls
        eval_model_list = data.eval_args.get("eval_model_list", [])
        available_eval_models = [
            model for model in eval_model_list if model in self.available_models
        ]
        if not available_eval_models:
            available_eval_models = self.available_models

        total_eval_calls = (
                len(all_conversations) * len(available_eval_models) * max_eval_try
        )

        # Stage 2: Direct parallel evaluation of all conversations
        print(
            f"Stage 2: Evaluating {len(all_conversations)} conversations with {len(available_eval_models)} models, "
            f"{max_eval_try} tries each = {total_eval_calls} total evaluations using {eval_batch_size} workers..."
        )

        # Prepare evaluation tasks for all conversations directly
        evaluation_tasks = []
        for idx, conversation_data in enumerate(all_conversations):
            evaluation_tasks.append(
                (
                    conversation_data,
                    data.eval_args,
                    max_eval_try,
                    idx,  # conversation index
                )
            )

        # Execute all evaluations in parallel
        all_scores = []
        all_confidences = []
        all_reasoning = []

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

        # Process all results
        for result in evaluation_results:
            if result["scores"]:
                # Average scores for this conversation across all models/tries
                conv_avg_score = sum(result["scores"]) / len(result["scores"])
                all_scores.append(conv_avg_score)

            if result["confidences"]:
                conv_avg_confidence = sum(result["confidences"]) / len(
                    result["confidences"]
                )
                all_confidences.append(conv_avg_confidence)

            if result["reasoning"]:
                all_reasoning.extend(result["reasoning"])

        # Calculate overall averages across all conversations
        raw_avg_score = sum(all_scores) / len(all_scores) if all_scores else 5.0
        overall_avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else None
        )
        combined_reasoning = "\n\n".join(all_reasoning) if all_reasoning else None

        # Normalize the final score to (0, 1) range
        overall_avg_score = self._normalize_score(raw_avg_score)
        print(f"Overall normalized score (0-1 range): {overall_avg_score:.4f}")
        score_finally = overall_avg_score * overall_avg_confidence
        return LLMJudgeMetrics(
            score=score_finally
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
