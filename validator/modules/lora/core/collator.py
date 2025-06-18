from typing import Any, Dict, List

import torch
from loguru import logger


class SFTDataCollator(object):
    """Collates samples for supervised fine-tuning style causal language modelling.

    It pads/truncates each sequence in the batch to the same length and builds the
    labels tensor such that only tokens corresponding to the assistant response
    are used for the loss (i.e. tokens whose `target_mask` == 1).
    """

    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        lengths = [len(x["input_ids"]) for x in batch if x["input_ids"] is not None]
        if not lengths:
            raise ValueError("Empty batch received by SFTDataCollator")
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            target_mask = x["target_mask"]
            if input_ids is None:
                logger.warning("Sample with None input_ids encountered – skipping")
                continue

            padding_len = batch_max_len - len(input_ids)
            # pad
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate (in case the sample is longer than max_seq_length)
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        # we only compute loss on tokens coming from the assistant
        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        } 