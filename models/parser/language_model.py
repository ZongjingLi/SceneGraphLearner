import torch
import torch.nn as nn

class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = 0
        self.encoder = 0

    def forward(self, x):
        return x