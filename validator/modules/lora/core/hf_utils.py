from huggingface_hub import HfApi
from loguru import logger

api = HfApi()


def download_lora_config(repo_id: str, revision: str = "main") -> bool:
    """Attempt to download `adapter_config.json` from a LoRA repository.

    Returns `True` if the file is found, `False` otherwise.
    """
    try:
        api.hf_hub_download(
            repo_id=repo_id,
            filename="adapter_config.json",
            local_dir="lora",
            revision=revision,
        )
    except Exception as e:
        if "adapter_config.json" in str(e):
            logger.info("No adapter_config.json found – treating repo as full model")
            return False
        raise  # propagate unexpected errors
    return True


def download_lora_repo(repo_id: str, revision: str = "main") -> None:
    """Download the entire LoRA repository snapshot to the local `lora` directory."""
    api.snapshot_download(repo_id=repo_id, local_dir="lora", revision=revision) 