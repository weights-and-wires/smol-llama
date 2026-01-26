"""Learning rate scheduling utilities."""
import math


def get_cosine_lr(
    step: int,
    lr: float,
    warmup_iters: int,
    max_iters: int,
    min_lr_ratio: float = 0.1
) -> float:
    """
    Calculate learning rate using linear warmup + cosine decay schedule.

    Args:
        step: Current training step.
        lr: Base/peak learning rate.
        warmup_iters: Number of warmup iterations.
        max_iters: Total number of training iterations.
        min_lr_ratio: Minimum LR as a ratio of peak LR (default: 0.1).

    Returns:
        The learning rate for the current step.
    """
    # Linear warmup phase
    if step < warmup_iters:
        return lr * step / warmup_iters

    # Cosine decay phase
    decay = (step - warmup_iters) / (max_iters - warmup_iters)
    min_lr = lr * min_lr_ratio
    return min_lr + (1 - min_lr_ratio) * lr * 0.5 * (1 + math.cos(math.pi * decay))


def update_lr(optimizer, lr: float) -> None:
    """
    Update learning rate for all parameter groups in the optimizer.

    Args:
        optimizer: PyTorch optimizer.
        lr: New learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

