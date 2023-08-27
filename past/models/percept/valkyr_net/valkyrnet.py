from re import L
import torch
import torch.nn as nn

class ValkyrNet(nn.Module):
    def __init__(self, config):
        super().__init__()
    
    def forward(self,x):
        return x