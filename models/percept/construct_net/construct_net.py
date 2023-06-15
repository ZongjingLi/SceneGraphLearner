import torch
import torch.nn as nn
import torch.nn.functional as F

from .convs import *
from .projection import *
from .graph_propagation import *

from types import SimpleNamespace

class ConstructNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        # [Convolution on the Image Grid]
        node_feat_size = config.node_feat_size
        self.grid_conv = RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))

        # [Construct Quarters]

    def forward(self, x):
        conv_features = self.grid_conv(x)
        return x

class ConstuctLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nodes, edge_affinities):
        pass