from validator.exceptions import RecoverableException
from validator.modules.base import BaseValidationModule, BaseConfig, BaseInputData, BaseMetrics

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

    def validate(self, data: LoRAInputData, **kwargs) -> LoRAMetrics:
        """Run the validation procedure.

        The logic here is adapted from `validator/modules/lora/src/validate.py` but
        pared-down so that it only performs the forward pass and metric
        computation needed for the validator. All networking / ledger interaction
        has been removed.
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
        val_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            fp16=self.config.fp16,
            remove_unused_columns=self.config.remove_unused_columns,
            do_train=False,
            do_eval=True,
            report_to=[],  # silence wandb etc.
        )

        model_repo = data.hg_repo_id
        revision = data.revision or "main"
        tokenizer_repo = model_repo  # may change if we detect LoRA

        # ------------------------------------------------------------------
        # Determine if the repo contains LoRA adapter weights or a full model
        # ------------------------------------------------------------------
        is_lora = download_lora_config(model_repo, revision)
        if is_lora:
            adapter_cfg_path = Path("lora/adapter_config.json")
            try:
                with open(adapter_cfg_path, "r") as f:
                    adapter_cfg = json.load(f)
                base_model_path = adapter_cfg.get("base_model_name_or_path")
                if base_model_path:
                    tokenizer_repo = base_model_path
            except Exception as e:
                logger.warning(f"Failed to parse adapter_config.json: {e}")
        
        # ------------------------------------------------------------------
        # Tokeniser
        # ------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ------------------------------------------------------------------
        # Evaluation dataset
        # ------------------------------------------------------------------
        template_name = data.base_model if data.base_model in template_dict else "default"
        template = template_dict[template_name]
        eval_dataset = UnifiedSFTDataset(
            file=data.eval_file,
            tokenizer=tokenizer,
            max_seq_length=data.context_length,
            template=template,
        )

        total_bytes, total_target_tokens = calculate_bytes_and_tokens(
            eval_dataset, tokenizer, logger
        )
        token_byte_ratio_value = get_token_byte_ratio(total_target_tokens, total_bytes)

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
        model = _load_model()

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
        data_collator = SFTDataCollator(tokenizer, max_seq_length=data.context_length)
        trainer = Trainer(
            model=model,
            args=val_args,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        logger.info("Running evaluation …")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("nan"))

        # ------------------------------------------------------------------
        # Compute derived metrics
        # ------------------------------------------------------------------
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