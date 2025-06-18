import math
import numbers
from loguru import logger

__all__ = [
    "calculate_bpc_bppl_metrics",
    "get_token_byte_ratio",
    "calculate_bytes_and_tokens",
]


def calculate_bpc_bppl_metrics(eval_loss, total_target_tokens, total_bytes):
    """Calculate BPC (bits per character) and related metrics.

    Args:
        eval_loss (float): Average token-level loss in *nats* (ln P^{-1}).
        total_target_tokens (int): Number of target tokens in evaluation set.
        total_bytes (int): Number of target bytes in evaluation set.

    Returns:
        dict: Metrics dictionary compatible with `LoRAMetrics` schema.
    """
    if (
        total_bytes == 0
        or not isinstance(eval_loss, numbers.Real)
        or math.isnan(eval_loss)
        or math.isinf(eval_loss)
    ):
        logger.warning("Invalid inputs for BPC computation – returning infinities.")
        return {
            "bpc": float("inf"),
            "bppl": float("inf"),
            "nll_token_nats_total": float("nan"),
            "nll_token_bits_total": float("nan"),
        }

    nll_token_nats_total = eval_loss * total_target_tokens
    nll_token_bits_total = nll_token_nats_total / math.log(2)
    bpc = nll_token_bits_total / total_bytes

    if math.isinf(bpc):
        bppl = float("inf")
    else:
        try:
            bppl = math.pow(2, bpc)
        except OverflowError:
            bppl = float("inf")

    return {
        "bpc": bpc,
        "bppl": bppl,
        "nll_token_nats_total": nll_token_nats_total,
        "nll_token_bits_total": nll_token_bits_total,
    }


def get_token_byte_ratio(total_target_tokens, total_bytes):
    """Token / byte ratio used for sanity checks."""
    if total_bytes == 0:
        return float("inf")
    return total_target_tokens / total_bytes


def calculate_bytes_and_tokens(eval_dataset, tokenizer, logger):
    """Iterate over `eval_dataset` to compute target bytes and tokens."""
    total_bytes = 0
    total_target_tokens = 0
    logger.info("Calculating total bytes and tokens of targets in evaluation set …")

    for item in eval_dataset:
        target_ids = [
            tid for tid, m in zip(item["input_ids"], item["target_mask"]) if m == 1
        ]
        if target_ids:
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
            total_bytes += len(target_text.encode("utf-8"))
            total_target_tokens += len(target_ids)

    return total_bytes, total_target_tokens 