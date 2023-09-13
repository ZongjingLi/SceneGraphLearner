import torch
import torch.nn as nn

import numpy as np

class NeuroSymbolicPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pseudo_param = nn.Parameter(torch.randn(3,3))

    def get_action(self, obs):
        loss = self.pseudo_param.norm()
        action = np.random.randint(0,6)
        return action, loss