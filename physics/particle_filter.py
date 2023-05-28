import torch
import torch.nn as nn

import numpy as np

class ParticleFilter(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, world_state, belief):
        pass