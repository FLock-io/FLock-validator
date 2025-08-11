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


class LLMJudgeException(RecoverableException):

    pass


class LLMJudgeConfig(BaseConfig):
    temperature: float = 0.7
    max_tries: int = 3


class LLMJudgeMetrics(BaseMetrics):
    score: float  # LLM output score as the main metric
    confidence: Optional[float] = None  # Optional confidence measure
    reasoning: Optional[str] = None  # Optional reasoning from LLM


class LLMJudgeInputData(BaseInputData):
    task_id: int
    test_data_url: str
    evaluation_criteria: Optional[str] = None


class LLMJudgeValidationModule(BaseValidationModule):

    config_schema = LLMJudgeConfig
    metrics_schema = LLMJudgeMetrics
    input_data_schema = LLMJudgeInputData
    task_type = "llm_evaluation"

    def __init__(self, config: LLMJudgeConfig, **kwargs):
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

    def _call_gpt(self, messages: List[Dict[str, str]], model: str) -> str:
        params = {
            "model": model,
            "messages": messages,
            "temperature": self.config.temperature,
            "seed": random.randint(0, 10000),
        }

        try:
            completion = self.client.chat.completions.create(**params)
            return completion.choices[0].message.content
        except Exception as e:
            raise LLMJudgeException(f"API call failed: {e}")

    def _load_jsonl_conversations(self, data_url: str) -> List[Dict[str, Any]]:

        response = requests.get(data_url)
        response.raise_for_status()

        conversations = []
        for line_num, line in enumerate(response.text.strip().split("\n"), 1):
            if line.strip():  # Skip empty lines
                try:
                    json_data = json.loads(line)
                    conversations.append(json_data)
                except json.JSONDecodeError as e:
                    continue

        if not conversations:
            raise LLMJudgeException("No valid JSON conversations found in JSONL file")

        return conversations

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

        Returns:
            LLMJudgeMetrics: Metrics containing the averaged LLM evaluation score across all conversations
        """

        all_conversations = self._load_jsonl_conversations(data.test_data_url)

        conversation_scores = []
        conversation_confidences = []
        all_conversation_reasoning = []

        # Process each conversation
        for conv_idx, conversation_data in enumerate(all_conversations):
            conversation_context = self._format_single_conversation(conversation_data)
            messages = self._construct_evaluation_prompt(
                conversation_context, data.task_id
            )

            conv_scores = []
            conv_confidences = []
            conv_reasoning = []

            for try_num in range(self.config.max_tries):
                if not self.available_models:
                    raise LLMJudgeException("No available models found")
                selected_model = random.choice(self.available_models)
                response = self._call_gpt(messages, selected_model)
                parsed_result = self._parse_llm_response(response)

                conv_scores.append(parsed_result["score"])
                if parsed_result["confidence"] is not None:
                    conv_confidences.append(parsed_result["confidence"])
                if parsed_result["reasoning"]:
                    conv_reasoning.append(
                        f"Conv{conv_idx + 1}-Try{try_num + 1}({selected_model}): {parsed_result['reasoning']}"
                    )

            # Calculate average for single conversation
            if conv_scores:
                conv_avg_score = sum(conv_scores) / len(conv_scores)
                conversation_scores.append(conv_avg_score)

                if conv_confidences:
                    conv_avg_confidence = sum(conv_confidences) / len(conv_confidences)
                    conversation_confidences.append(conv_avg_confidence)

                if conv_reasoning:
                    all_conversation_reasoning.extend(conv_reasoning)

            else:
                print(
                    f"Warning: No valid scores obtained for conversation {conv_idx + 1}"
                )

        # Calculate overall averages across all conversations
        overall_avg_score = sum(conversation_scores) / len(conversation_scores)
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

        print(
            f"Overall average score across {len(conversation_scores)} conversations: {overall_avg_score:.2f}"
        )

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


MODULE = LLMJudgeValidationModule
