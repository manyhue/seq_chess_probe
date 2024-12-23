from dataclasses import dataclass

import torch
from lib.modules import ClassifierModule
import torch.nn as nn

from lib.optim import WarmupCosineLR
from lib.train import Trainer
from lib.utils import Config


@dataclass(kw_only=True)
class ProbeConfig(Config):
    n_classes: int = 1024


class LinearProbe(ClassifierModule):
    def __init__(self, model, target_layer, c: ProbeConfig):
        self.save_config(c)
        super().__init__()
        self.model = model
        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # Add a linear layer for classification
        # Impossible to infer output of target_layer without forward?
        self.probe_fc = nn.LazyLinear(c.n_classes)

        self.activations = None

        def hook_fn(m, i, o):
            self.activations = o.detach().view(o.shape[0], -1)

        self.hook_handle = target_layer.register_forward_hook(hook_fn)

    def forward(self, idx):
        _ = self.model(idx)
        output = self.probe_fc(self.activations)
        return output


class ProbeTrainer(Trainer):
    def prepare_optimizers(self):
        self.optim = torch.optim.Adam(self.model.probe_fc.parameters())
        self.sched = WarmupCosineLR(
            self.optim,
            warmup_iters=200,
            min_lr=6e-6,
            lr_decay_iters=10000,
            lr=self.lr,
        )
