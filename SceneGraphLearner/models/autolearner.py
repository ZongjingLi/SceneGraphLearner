from .scenelearner import SceneLearner
from .physics import *
from .parser import *
from .planning import *

import torch
import torch.nn as nn

class AutoLearner(nn.Module):
    def __init__(self, config):
        super().__init__()
        # [Scene Graph Construction Module]
        self.scenelearner = SceneLearner(config)

        # [Intuitive Physics Module]
        self.particle_filter = NeuroParticleFilter(config)

        # [Planning Abstraction Module]
        self.planning_model = NeuroSymbolicPlanner(config)

    def forward(self, x):
        return x

    def display(self): return