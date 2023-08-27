import torch
import torch.nn as nn
from types import SimpleNamespace

from .utils import *
from .convs import *


class GNNSoftPooling(nn.Module):
    def __init__(self, output_node_num = 10):
        super().__init__()
        
    
    def forward(self, x):
        return x

class ObjectRender(nn.Module):
    def __init__(self,):
        super().__init__()

class ValkyrNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.device
        # construct the grid domain connection
        self.imsize = config.imsize
        self.perception_size = config.perception_size
        # build the connection graph for the grid domain
        self.spatial_coords = grid(self.imsize,self.imsize,device=device)
        self.spatial_edges =  build_perception(self.imsize,self.perception_size,device = device)

        # [Grid Convs]
        conv_feature_dim = config.conv_feature_dim
        self.grid_convs =RDN(SimpleNamespace(G0=conv_feature_dim  ,RDNkSize=3,n_colors=3,RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        
        # [Diff Pool Construction]
        hierarchy_nodes = config.hierarchy_construct
        self.diff_pool = nn.ModuleList([
            GNNSoftPooling(output_node_num = node_num ) for node_num in hierarchy_nodes
        ])

        # [Render Fields]
        self.render_fields = nn.ModuleList([])

        self.conv2object_feature = None
    
    def forward(self,x):
        
        return x