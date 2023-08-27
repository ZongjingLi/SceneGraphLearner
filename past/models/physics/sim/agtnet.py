import torch
import torch.nn as nn
import torch.nn.functional as F

class ParticleEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.linear0 = nn.Linear(input_dim,     hidden_dim)
        self.linear1 = nn.Linear(hidden_dim,    hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,    output_dim)
        self.activate = nn.ReLU()
        self.output_dim = output_dim

    def forward(self, x):
        # B,N,D1 -> B,N,D2
        B,N,D = x.shape
        x = x.reshape(B * N, D)
        x = self.linear0(x)
        x = self.activate(x)
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        x = self.activate(x)
        x = x.reshape(B, N, self.output_dim)
        return x

class ParticlePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super().__init__()
        self.output_dim = output_dim
        self.linear0 = nn.Linear(input_dim,     hidden_dim)
        self.linear1 = nn.Linear(hidden_dim,    output_dim)
        self.activate = nn.ReLU()

    def forward(self, x):
        # B,N,D1 -> B,N,D2
        B,N,D = x.shape 
        x = x.reshape([B * N, D])
        x = self.linear0(x)
        x = self.activate(x)
        x = self.linear1(x)
        return x.reshape(B,N,self.output_dim)

class RelationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.linear_0 = nn.Linear(input_dim,  hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        B ,N, Z = x.shape
        x = x.reshape(B * N, Z)
        x = F.relu(self.linear_0(x))
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x.reshape(B, N, self.output_dim)

class Propagator(nn.Module):
    def __init__(self, input_dim, output_dim, residual = False):
        super().__init__()
        self.residual = residual
        self.output_dim = output_dim
        self.linear_0 = nn.Linear(input_dim, output_dim)

    def forward(self, inputs, res = None):
        B, N, D = inputs.shape
        if self.residual:
            x = self.linear_0(inputs.reshape(B * N, D))
            x = F.relu(x + res.reshape(B * N, self.output_dim))
        else:
            x = F.relu(self.linear_0(inputs.reshape(B * N, D)))
        return x.reshape(B, N, self.output_dim)

class PropModule(nn.Module):
    def __init__(self, config, input_dim, output_dim, batch = True, residual = False):
        super().__init__()
        device = config.device
        self.device = device
        self.config = config
        self.batch  = batch

        relation_dim = config.relation_dim

        pfd = config.prop_feature_dim

        particle_feature_dim = config.particle_feature_dim
        relation_feature_dim = config.relation_feature_dim
        prop_feature_dim = config.prop_feature_dim
        pfd = prop_feature_dim

        self.residual = residual
        self.effect_dim = config.prop_feature_dim

        self.particle_encoder = ParticleEncoder(input_dim,pfd, particle_feature_dim)
        self.relation_encoder = RelationEncoder(2*input_dim + relation_dim,\
                                             relation_feature_dim, relation_feature_dim)
        self.particle_propagator = Propagator(2*pfd, pfd, self.residual)
        self.relation_propagator = Propagator(2*pfd+relation_feature_dim, pfd)
        self.particle_predictor  = ParticlePredictor(pfd,output_dim,pfd)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, state, Rr, Rs, Ra, steps):
        # [Encode Particle Features]
        particle_effect = torch.autograd.Variable(\
            torch.zeros(state.shape[0],state.shape[1], self.effect_dim))
        particle_effect = particle_effect.to(self.device)

        if self.batch:
            Rrp = torch.transpose(Rr, 1, 2)
            Rsp = torch.transpose(Rs, 1, 2)
            state_r = Rrp.bmm(state)
            state_s = Rsp.bmm(state)
        else:
            print("Oh come on, why not use the batch-wise operation")
        # [Particle Encoder]
        particle_encode = self.particle_encoder(state)

        # [Relation Encoder] calculate the relation encoding
        #print(state_r.shape,state_s.shape, Ra.shape)
        relation_encode = self.relation_encoder(
            torch.cat([state_r, state_s, Ra], 2)
        )

        for i in range(steps):
            if self.batch:
                effect_r = Rrp.bmm(particle_effect)
                effect_s = Rsp.bmm(particle_effect)
            else: pass

            # calculate relation effect
            relation_effect = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2)
            )

            # calculate particle effect by aggregating relation effect
            if self.batch:
                effect_agg = Rr.bmm(relation_effect)
            
            # calculate particle effect
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_agg], 2),
                res = particle_effect
            )

        pred = self.particle_predictor(particle_effect)

        return pred

class AgtNet(nn.Module):
    def __init__(self, config, residual = False):
        super().__init__()
        
        position_dim = config.position_dim
        state_dim = config.state_dim
        attr_dim = config.attr_dim
        action_dim = config.action_dim
        particle_fd = config.particle_feature_dim
        relation_fd = config.relation_feature_dim
        observation = config.observation

        if observation == "partial":
            batch = False
            input_dim = state_dim
    
        if observation == "full":
            batch = True
            input_dim = state_dim + action_dim + attr_dim
            self.model = PropModule(config, input_dim, position_dim, batch, residual)

    def encode(self, data, steps):
        state, Rr, Rs, Ra = data
        return self.encoder(state, Rr, Rs, Ra, steps)

    def decode(self, data, steps):
        state, Rr, Rs, Ra = data
        return self.decoder(state, Rr, Rs, Ra, steps)

    def rollout(self, state, action):
        # used only for partially observable case
        return self.roller(torch.cat([state, action], 2))
    
    def to_latent(self, state, mode = "sum"):
        if mode == "sum":
            return torch.sum(state, 1, keepdim = True)
        elif mode == "mean":
            return torch.mean(state, 1, keepdim = True)
        raise AssertionError("Unsupported Aggregation Function")
    
    def forward(self, data, steps, action = None):
        # used only for fully observable case
        state, Rr, Rs, Ra = data
        if action is not None:
            state = torch.cat([state, action], 2)
        else:
            state = state
        return self.model(state, Rr, Rs, Ra, steps)