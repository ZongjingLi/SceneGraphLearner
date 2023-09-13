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
        while not done or steps < max_steps:
            local_obs, global_obs = env.render()
            action = model.get_action(local_obs)
            update = env.step(action)

            reward = update["reward"]
            done = update["done"]

            if visualize_map:
                plt.subplot(121)
                plt.cla();plt.axis("off")
                plt.imshow(local_obs)
                plt.subplot(122)
                plt.cla();plt.axis("off")
                plt.imshow(global_obs)
                plt.pause(0.01)
            steps += 1
    return 0

if __name__ == "__main__":
    env = Northrend(config)
    random_model = RandomModel(6)

    simulate_env(random_model, env,)