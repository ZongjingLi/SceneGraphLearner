import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import *
from .projection import *
from .graph_propagation import *

class ConstructNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # [Convolution on the Image Grid]
        self.grid_conv = 0

    def forward(self, x):
        return x

class ConstuctLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nodes, edge_affinities):
        pass