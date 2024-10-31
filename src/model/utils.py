"""
Utility functions
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from model.transformers import TransformerModel


# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
                                    num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps + 1))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_openai_lr(transformer_model: TransformerModel):
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)


def torch_nanmean(x, dim=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(dim=dim)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(dim=dim)
    if return_nanshare:
        return value / num, 1.-num/x.shape[dim]
    return value / num
