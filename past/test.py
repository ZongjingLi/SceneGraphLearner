
from models import *
from config import *

model = AutoLearner(config)

print(model)

B = 2
T = 32
N = 11
D = config.state_dim + config.attr_dim
rDim = config.relation_dim

states = torch.randn([B,N,D])
Rr = torch.ones([B,N,N])
Rs = torch.ones([B,N,N])
Ra = torch.ones([B,N,rDim])

data = states, Rr, Rs, Ra
outputs = model.particle_filter.stepper(data, steps = 5, action = None)

print(outputs.shape)