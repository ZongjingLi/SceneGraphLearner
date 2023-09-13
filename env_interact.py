from this import d
import torch.nn as nn
import torch

import numpy as np
from env.mkgrid.northrend_env import Northrend
from config import *

class RandomModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n_actions = n
    
    def get_action(self, obs):
        return np.random.randint(0,self.n_actions)

def simulate_env(model, env, goal = None, max_steps = 1000, visualize_map = True):
    for epoch in range(1):
        done = False
        steps = 0
        env.reset()

        # [Build a Plan]
        if goal is not None:model.plan(goal)
        plt.figure("epoch:{}".format(epoch))
        while not done or steps:
            obs = env.get_observation()
            action = model.get_action(obs)
            next_state = 0

            if visualize_map:
                plt.imshow()
                plt.pause(0.01)
    return 0

if __name__ == "__main__":
    env = Northrend(config)
    random_model = RandomModel(4)