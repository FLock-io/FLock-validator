from validator.exceptions import RecoverableException
from validator.modules.base import BaseValidationModule, BaseConfig, BaseInputData, BaseMetrics
from constant import SUPPORTED_BASE_MODELS

# When raised, the assignment won't be marked as failed automatically and it will be retried after the user
# fixes the problem and restarts the process.
class InvalidConfigValueException(RecoverableException):
    pass


class LoRAConfig(BaseConfig):
    per_device_eval_batch_size: int
    fp16: bool
    output_dir: str
    remove_unused_columns: bool

class LoRAMetrics(BaseMetrics):
    loss: float
    bpc: float
    bppl: float
    nll_token_nats_total: float
    nll_token_bits_total: float

class LoRAInputData(BaseInputData):
    hg_repo_id: str
    revision: str
    base_model: str
    context_length: int
    max_params: int
    validation_set_url: str
    gpu_type: str | None = None

class LoRAValidationModule(BaseValidationModule):
    config_schema = LoRAConfig
    metrics_schema = LoRAMetrics
    input_data_schema = LoRAInputData
    task_type = "training"

    def __init__(self, config: LoRAConfig, **kwargs):
        # Store the config for later use
        self.config = config
    
    def validate_config(self):
        """Validate logical correctness of LoRA config values."""
        import os
        
        if self.config.per_device_eval_batch_size <= 0:
            raise InvalidConfigValueException("`per_device_eval_batch_size` must be > 0.")
        if not self.config.output_dir.strip():
            raise InvalidConfigValueException("`output_dir` must not be empty or blank.")
        if not os.path.isdir(self.config.output_dir):
            raise InvalidConfigValueException(f"`output_dir` '{self.config.output_dir}' does not exist or is not a directory.")


    def validate(self, data: LoRAInputData, **kwargs) -> LoRAMetrics:
        """Run the validation procedure.

        The logic here is adapted from `validator/modules/lora/src/validate.py` but
        pared-down so that it only performs the forward pass and metric
        computation needed for the validator. All networking / ledger interaction
        has been removed.
        
        For error handling, we use `InvalidConfigValueException` instead of simply returning as
        was previously done. Any errors will be bubbled up to the validation runner.
        """
        import json
        import math
        import os
        import gc
        from pathlib import Path

        import torch
        from loguru import logger
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import PeftModel
        from json import JSONDecodeError

        from .core import (
            SFTDataCollator,
            UnifiedSFTDataset,
            template_dict,
            calculate_bpc_bppl_metrics,
            calculate_bytes_and_tokens,
            get_token_byte_ratio,
            download_lora_config,
            download_lora_repo,
            _log_summary_table,
        )

        # ------------------------------------------------------------------
        # Build HF `TrainingArguments` object from config
        # ------------------------------------------------------------------
        # Validate config values before creating TrainingArguments (should be done in __init__ ideally)
        logger.info("Validating config...")
        try:
            self.validate_config()
        except InvalidConfigValueException as e:
            logger.error(f"Invalid config: {e}")
            raise InvalidConfigValueException(f"Config validation failed: {e}")
        logger.info("Config validation passed.")
        
        # Enclose in try-except to catch any issues with TrainingArguments creation because there might be issues with the config values
        try:
            logger.info("Creating TrainingArguments...")
            val_args = TrainingArguments(
                output_dir=self.config.output_dir,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                fp16=self.config.fp16,
                remove_unused_columns=self.config.remove_unused_columns,
                do_train=False,
                do_eval=True,
                report_to=[],  # silence wandb etc.
            )
        except Exception as e:
            logger.error(f"Failed to create TrainingArguments: {e}")
            raise InvalidConfigValueException(f"TrainingArguments creation failed: {e}") # Might be good to introduce more exceptions for specific cases

        model_repo = data.hg_repo_id
        revision = data.revision or "main"
        tokenizer_repo = model_repo  # may change if we detect LoRA

        # ------------------------------------------------------------------
        # Determine if the repo contains LoRA adapter weights or a full model
        # ------------------------------------------------------------------
        # Add more robust error handling for LoRA detection
        is_lora = download_lora_config(model_repo, revision)

        if is_lora:
            adapter_cfg_path = Path("lora/adapter_config.json")
            if not adapter_cfg_path.exists():
                    # is_lora is True, but adapter_config.json does not exist
                    logger.error(
                    f"Model {model_repo} is identified as LoRA, but its adapter_config.json was not downloaded or found at {adapter_config_path}. "
                    f"This could be due to an issue with 'download_lora_config' or the repository structure for the LoRA model. "
                    )
                    raise InvalidConfigValueException(f"adapter_config.json not found for LoRA model {model_repo}. ")
            else:
                logger.info(
                    f"Model {model_repo} is a LoRA model. Validating its base model for tokenizer."
                )
                try:
                    with open(adapter_cfg_path, "r") as f:
                        adapter_cfg = json.load(f)

                    base_model_path = adapter_cfg.get("base_model_name_or_path")
                    if not base_model_path or not base_model_path.strip():
                        logger.error(
                            f"LoRA model {model_repo} does not specify 'base_model_name_or_path' "
                        )
                        return # exit function early 
                    if base_model_path not in SUPPORTED_BASE_MODELS: # need to define SUPPORTED_BASE_MODELS
                        logger.error(
                            f"LoRA's base model '{base_model_path}' is not in SUPPORTED_BASE_MODELS. "
                        ) 
                    tokenizer_repo = base_model_path

                except (FileNotFoundError, JSONDecodeError) as e: # case where adapter_config.json is missing is already handled
                    logger.error(f"Failed to read adapter_config.json: {e}")
                    raise InvalidConfigValueException(f"adapter_config.json parsing failed: {e}")
                except InvalidConfigValueException as e:
                    logger.error(str(e))
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error reading adapter_config.json: {e}")
                    raise
        else:
            logger.info(
                f"Model {model_repo} is not identified as a LoRA model. "
                f"Using its own path for tokenizer: {model_repo}."
            )
        # ------------------------------------------------------------------
        # Tokeniser
        # ------------------------------------------------------------------
        # Enclose the tokenizer loading in a try-except block to handle potential errors
        def load_tokenizer(model_name_or_path: str) -> AutoTokenizer: # move this to a separate file and import it
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                use_fast=True,
            )
            if "gemma" in model_name_or_path.lower():
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"]}
                )

            if tokenizer.__class__.__name__ == "QWenTokenizer":
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.eod_id
                tokenizer.eos_token_id = tokenizer.eod_id
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
            assert tokenizer.eos_token_id is not None, "eos_token_id should not be None"
            logger.info(f"vocab_size of tokenizer: {tokenizer.vocab_size}")
            return tokenizer
        
        try:
            logger.info("Loading tokenizer …")
            tokenizer = load_tokenizer(tokenizer_repo)
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_repo}: {e}")
            raise InvalidConfigValueException(f"Tokenizer loading failed: {e}")

        # ------------------------------------------------------------------
        # Evaluation dataset
        # ------------------------------------------------------------------
        
        template_name = data.base_model if data.base_model in template_dict else "default"
        template = template_dict[template_name]
        
        def load_sft_dataset(eval_file: str, max_seq_length: int, template_name: str, tokenizer: AutoTokenizer) -> UnifiedSFTDataset:
            if template_name not in template_dict.keys():
                raise ValueError(
                    f"template_name doesn't exist, all template_name: {template_dict.keys()}"
                )
            template = template_dict[template_name]
            logger.info("Loading data with UnifiedSFTDataset")
            return UnifiedSFTDataset(eval_file, tokenizer, max_seq_length, template)
        
        try:
        # Possible errors here: data.eval_file not found, etc.
            eval_dataset = load_sft_dataset(
                eval_file=data.eval_file,
                max_seq_length=data.context_length,
                template_name=template,
                tokenizer=tokenizer
                )
    
            total_bytes, total_target_tokens = calculate_bytes_and_tokens(eval_dataset, tokenizer, logger)
            if total_bytes == 0:
                logger.warning(
                    "Total bytes in the evaluation dataset is 0. Cannot calculate BPC. Check dataset processing."
                )
            else:
                logger.info(f"Total target tokens (T): {total_target_tokens}")
                logger.info(f"Total target bytes (B): {total_bytes}")
                token_byte_ratio_value = get_token_byte_ratio(
                    total_target_tokens, total_bytes
                )
                logger.info(f"Token/Byte ratio (T/B): {token_byte_ratio_value:.4f}")
                if token_byte_ratio_value < 0.1:
                    logger.warning(
                        f"Token/Byte ratio ({token_byte_ratio_value:.4f}) is unusually low. Potential manipulation detected."
                    )
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset from {data.eval_file}: {e}")
            raise InvalidConfigValueException(f"Evaluation dataset loading failed: {e}")
        
        # ------------------------------------------------------------------
        # Model loading helper
        # ------------------------------------------------------------------
        def _load_model() -> AutoModelForCausalLM:
            dtype = torch.float32 if not torch.cuda.is_available() else (torch.float16 if val_args.fp16 else torch.bfloat16)
            model_kwargs = dict(trust_remote_code=True, torch_dtype=dtype, use_cache=False, device_map=None)

            if is_lora:
                # download all LoRA files first so PEFT can find them locally
                download_lora_repo(model_repo, revision)
                with open("lora/adapter_config.json", "r") as f:
                    adapter_config = json.load(f)
                base_model = adapter_config["base_model_name_or_path"]
                base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
                model = PeftModel.from_pretrained(base, "lora", device_map=None)
                model = model.merge_and_unload()
                return model
            else:
                return AutoModelForCausalLM.from_pretrained(model_repo, **model_kwargs)

        logger.info("Loading model …")
        try:
            model = _load_model()
        except Exception as e:
            logger.error(f"Exception occurred while loading model: {e}")
            raise InvalidConfigValueException(f"Model loading failed: {e}")

        if model is None:
            logger.error(f"Failed to load model from {model_repo}.")
            raise InvalidConfigValueException(f"Model loading failed: {model_repo}.")

        # Simple parameter count check
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params / 1e6:.2f} M")
        if total_params > data.max_params:
            logger.warning(
                f"Parameter count {total_params} exceeds limit {data.max_params}. Returning high loss metrics."
            )
            high_loss = 999.0
            return LoRAMetrics(
                loss=high_loss,
                bpc=high_loss,
                bppl=math.pow(2, high_loss),
                nll_token_nats_total=float("nan"),
                nll_token_bits_total=float("nan"),
            )

        # ------------------------------------------------------------------
        # Prepare trainer and run evaluation
        # ------------------------------------------------------------------
        try:
            data_collator = SFTDataCollator(tokenizer, max_seq_length=data.context_length)
            trainer = Trainer(
                model=model,
                args=val_args,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
        except Exception as e:
            logger.error(f"Failed to set up Trainer or data collator: {e}")
            raise InvalidConfigValueException(f"Trainer setup failed: {e}")

        try:
            logger.info("Running evaluation …")
            eval_metrics = trainer.evaluate()
            eval_loss = eval_metrics.get("eval_loss", float("nan"))
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise RuntimeError(f"Evaluation failed: {e}")

        # ------------------------------------------------------------------
        # Compute derived metrics
        # ------------------------------------------------------------------
        bpc_metrics = { # for failsafe scenario if total_bytes is 0
        "loss": float("nan"),   
        "bpc": float("inf"),
        "bppl": float("inf"),
        "nll_token_nats_total": float("nan"),
        "nll_token_bits_total": float("nan"),
        }
        if total_bytes > 0:
            bpc_metrics = calculate_bpc_bppl_metrics(
                eval_loss, total_target_tokens, total_bytes
            )

        _log_summary_table(
            model_name_or_path=model_repo,
            eval_loss=eval_loss,
            bpc_metrics=bpc_metrics,
            token_byte_ratio=token_byte_ratio_value,
            total_target_tokens=total_target_tokens,
            total_bytes=total_bytes,
            vocab_size=tokenizer.vocab_size,
            model_params_m=total_params / 1e6,
        )

        result = {
            "loss": eval_loss,
            "bpc": bpc_metrics["bpc"],
            "bppl": bpc_metrics["bppl"],
            "nll_token_nats_total": bpc_metrics["nll_token_nats_total"],
            "nll_token_bits_total": bpc_metrics["nll_token_bits_total"],
        }

        # ------------------------------------------------------------------
        # Clean-up GPU / CPU memory – important when running many validations
        # ------------------------------------------------------------------
        gc.collect()
        if model is not None:
            logger.debug("Offloading model to save memory")
            model.cpu()
            del model
        if eval_dataset is not None:
            logger.debug("Offloading eval_dataset to save memory")
            del eval_dataset
        torch.cuda.empty_cache()
        # purge temporary lora directory if it exists
        if os.path.exists("lora"):
            import shutil
            shutil.rmtree("lora", ignore_errors=True)

        return LoRAMetrics(**result)

    def cleanup(self):
        # Free resources
        pass

MODULE = LoRAValidationModule