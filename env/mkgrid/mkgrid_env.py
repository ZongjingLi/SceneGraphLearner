import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn

class MKGrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        H, W = config.env_global_resolution

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

    northrend_colors = [
    '#1f77b4',  # muted blue
    '#005073',  # safety orange
    '#96aab3',  # cooked asparagus green
    '#2e3e45',  # brick red
    '#08455e',  # muted purple
    '#575959',  # chestnut brown
    '#38677a',  # raspberry yogurt pink
    '#187b96',  # middle gray
    '#31393b',  # curry yellow-green
    '#1cd1ed'   # blue-teal
    ]
    from utils import *
    color_map_dict = {}
    save_json(color_map_dict)