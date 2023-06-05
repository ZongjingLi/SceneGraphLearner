import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_particles, input_size]
        Returns:
            [batch_size, n_particles, output_size]
        '''
        B, N, D = x.size()
        x = x.view(B * N, D)
        x = self.linear_1(self.relu(self.linear_0(x)))
        return x.view(B, N, self.output_size)

class ParticleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.D = output_size

    def forward(self, inputs):
        """
        inputs: particle states: [B, N, F]
        Batch, Num-Particle, Feature-Dim

        outputs: encoded states: [B, N, D]
        """
        D = self.D
        B, N, Z = inputs.shape
        x = inputs.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape([B,N,D])

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.D = output_size

    def forward(self, inputs):
        """
        inputs: relation states: [B, N, F]
        Batch, Num-relations, Feature-Dim

        outputs: encoded states: [B, N, D]
        """
        D = self.D
        B, N, Z = inputs.shape
        x = inputs.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape([B,N,D])

class Propagator(nn.Module):
    def __init__(self, input_size, output_size, residual = False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.residual = residual

        self.linear_0 = nn.Linear(input_size, output_size)

    def forward(self, inputs, res = None):
        B, N, D = inputs.size
        if self.residual:
            x = self.linear_0(inputs.reshape([B * N , D]))
            x = F.relu(x + res.reshape([B * N, self.output_size]))
        else:
            x = F.relu(self.linear_0(x.reshape(B * N, D)))
        return x.reshape([B, N, self.output_size])

class PropModule(nn.Module):
    def __init__(self, config, input_dim, output_dim, batch = True, residual = False):
        super().__init__()
        device = config.device

        self.device = device
        self.config = config

        self.batch = batch

        state_dim = config.state_dim
        attr_dim = config.attr_dim
        relation_dim = config.relation_dim
        action_dim = config.action_dim

        nf_particle = config.nf_particle
        nf_relation = config.nf_relation
        nf_effect = config.nf_effect

        self.nf_effect = nf_effect
        self.residual = residual

        # Particle Encoder
        self.particle_encoder = ParticleEncoder(
            input_dim, nf_particle, nf_effect
        )
        # Relation Encoder