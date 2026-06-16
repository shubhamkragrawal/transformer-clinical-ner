import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
) -> torch.Tensor:
    """
    Token classification loss with padding/subword-continuation masking.

    logits: (batch, seq_len, num_labels)
    labels: (batch, seq_len) — contains -100 at positions to ignore
             (padding, [CLS]/[SEP], non-first subwords)

    CrossEntropyLoss's ignore_index=-100 handles all masking automatically —
    this is exactly why -100 was chosen during label alignment in Day 1,
    it's PyTorch's built-in sentinel for "skip this position."
    """
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    # Flatten batch and seq_len dims: (batch*seq_len, num_labels) vs (batch*seq_len,)
    loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))
    return loss


def get_optimizer(model: nn.Module, config: dict) -> AdamW:
    t = config["training"]
    return AdamW(
        model.parameters(),
        lr=t["learning_rate"],
        weight_decay=t["weight_decay"],
    )


def get_scheduler(optimizer: AdamW, config: dict, num_training_steps: int) -> LambdaLR:
    """
    Linear warmup followed by linear decay to zero.
    Standard schedule for transformer training — warmup prevents early
    instability from large updates before AdamW's moment estimates settle.
    """
    warmup_steps = config["training"]["warmup_steps"]

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # Linear decay from 1.0 -> 0.0 over remaining steps
        remaining = num_training_steps - warmup_steps
        decayed = (num_training_steps - current_step) / max(1, remaining)
        return max(0.0, decayed)

    return LambdaLR(optimizer, lr_lambda)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """
    Clips gradient norm in-place to prevent exploding gradients.
    Returns the pre-clip norm so you can log/monitor it during training —
    spikes in this value often indicate a learning rate that's too high.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm.item()