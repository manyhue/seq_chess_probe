import inspect
import torch
from lib.optim import WarmupCosineLR
from lib.train import Trainer
from typing import override

from lib.utils import dbg


class GPTTrainer(Trainer):
    # from nanogpt
    @override
    def prepare_optimizer(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        self.optim = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )

        self.sched = WarmupCosineLR(
            self.optim,
            warmup_iters=200,
            min_lr=6e-7,
            lr_decay_iters=2000000,
            lr=self.lr,
        )


class SeqTrainer(GPTTrainer):
    @override
    def prepare_batch(self, batch):
        if self.gpus:
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            batch = batch[0].pin_memory().to(self.device, non_blocking=True)
            # batch = batch[0].to(self.device)
        else:
            batch = batch[0].to(self.device)
        return batch[:, :-1], batch[:, 1:]
