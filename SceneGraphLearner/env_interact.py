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
        return np.random.randint(0,self.n_actions), 0.0

def simulate_env(model, env, goal = None, max_steps = 1000, visualize_map = True):
    losses = []
    rewards = []
    for epoch in range(10):
        done = False
        steps = 0
        env.reset()

        # [Build a Plan]
        if goal is not None:model.plan(goal)
        plt.figure("epoch:{}".format(epoch))
        epoch_loss = 0
        epoch_reward = 0
        while not done and steps < max_steps:
            # [Get Current State]
            local_obs, global_obs = env.render()

            # [Action Bases on Current Plan and State]
            action,loss = model.get_action(local_obs)
            update = env.step(action)

            # [Calculate Reward and Epoch Loss]
            reward = update["reward"]
            done = update["done"]
            epoch_loss += reward
            epoch_loss += loss

            # [Visualize Results]
            if visualize_map:
                plt.subplot(121);plt.cla();plt.axis("off");plt.imshow(local_obs)
                plt.subplot(122);plt.cla();plt.axis("off");plt.imshow(global_obs)
                plt.pause(0.01)
            steps += 1
        losses.append(epoch_loss / steps)
        rewards.append(epoch_reward / steps)

    return losses, rewards


if __name__ == "__main__":
    env = Northrend(config)
    random_model = RandomModel(6)

    l,r = simulate_env(random_model, env, max_steps=100)
    print(l, r)