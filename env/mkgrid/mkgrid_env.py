import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

class MKGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        H, W = config.env_global_resolution = 0.0

        load_map = 0 # check if load default map
        self.grid = np.zeros([H,W])
        if load_map:
            return 0


    def render(self):
        return 0

    def step(self, action, state = None):
        if state is None:
            next_state = self.grid
        else: next_state = state
        done = 0
        outputs= {"prev_state":state,
                  "next_state":next_state,
                  "done":done}
        return outputs
    
if __name__ == "__main__":
    from utils import *
    color_map_dict = {
        "1":
    }
    save_json(color_map_dict)