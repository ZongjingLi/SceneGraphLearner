import torch
import torch.nn as nn

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

    def forward(self, x):
        return x