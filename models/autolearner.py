from .scenelearner import SceneLearner
from .physics import *
from .parser import *

import torch
import torch.nn as nn

class AutoLearner(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x