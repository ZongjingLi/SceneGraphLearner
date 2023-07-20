from logging import raiseExceptions
import torch
import torch.nn as nn
from .loss import *

class CSQ_Module(nn.Module):
    def __init__(self, config, num_slots):
        super().__init__()
    
    def forward(self,x):
        return x

class CSQNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        construct = (10,5)
        self.base_encoder = None
        self.csq_modules = []

    def forward(self, inputs):
        enc_in = inputs['point_cloud'] * self.scaling 
        query_points = inputs['coords'] * self.scaling 

        enc_in = torch.cat([enc_in, inputs['rgb']], 2)
        
        loss = {"reconstruction":1.0,"localization":0.1}
        outputs = {"loss":loss}
        return outputs