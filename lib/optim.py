import math
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_iters,
        lr_decay_iters,
        lr=None,
        min_lr=6e-5,
        last_epoch=-1,
    ):
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.lr = lr or self.base_lrs[0]
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, "deprecated")

    def get_lr(self):
        # Get the current iteration
        it = self.last_epoch

        # 1) Linear warmup phase
        if it < self.warmup_iters:
            lr_scale = it / self.warmup_iters
        # 2) If iteration > lr_decay_iters, return min learning rate
        elif it > self.lr_decay_iters:
            lr_scale = 0
        # 3) Cosine decay phase after warmup
        else:
            decay_ratio = (it - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

        # Update the learning rate for each parameter group
        return [self.min_lr + lr_scale * (self.lr - self.min_lr) for _ in self.base_lrs]
