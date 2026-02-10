# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
"""
Exponential Moving Average (EMA) wrapper for optimizers.
Adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
"""

import warnings
import torch
from torch.optim import Optimizer


class EMA:
    """
    EMA wrapper - does NOT inherit from Optimizer.
    Delegates all optimizer methods to the wrapped optimizer.
    """
    
    def __init__(self, opt, ema_decay=0.9999):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.0
        self.optimizer = opt

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)
        
        if not self.apply_ema:
            return retval

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                
                if "ema" not in state:
                    state["ema"] = p.data.clone()
                
                state["ema"].mul_(self.ema_decay).add_(p.data, alpha=1.0 - self.ema_decay)

        return retval

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def swap_parameters_with_ema(self, store_params_in_ema=True):
        if not self.apply_ema:
            return

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if "ema" not in self.optimizer.state[p]:
                    continue
                ema = self.optimizer.state[p]["ema"]
                if store_params_in_ema:
                    tmp = p.data.clone()
                    p.data.copy_(ema)
                    self.optimizer.state[p]["ema"] = tmp
                else:
                    p.data.copy_(ema)
