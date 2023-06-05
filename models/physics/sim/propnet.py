import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

    def forward(self, x):
        return 0

class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        """
        inputs: particle states: [B, N, Z]
        Batch, Particle-Num, Feature-Dim

        outputs: encoded states: [B, N, D]
        """
        B, N, Z = inputs.shape
        x = inputs.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

class PropModule(nn.Module):
    def __init__(self, config):
        super().__init__()