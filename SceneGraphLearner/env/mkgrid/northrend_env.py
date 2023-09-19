import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
from utils import *

class Northrend(nn.Module):
    def __init__(self, config, load_map = None):
        super().__init__()
        H, W = config.env_global_resolution    
        locH, locW = config.env_local_resolution

        self.global_resolution = (H,W)
        self.local_resolution = (locH,locW)
        # [Build Global Grid]
        base = (locW//2,locH//2)
        self.base = base
        self.grid = torch.tensor(np.zeros([H+locH,W+locW])).int()
        
        # [Load the Map]
        if load_map is not None:
            self.grid[base[0]:base[0]+W,base[1]:base[1]+H] = np.load(load_map)
        else:
            self.grid[base[0]:base[0]+W,base[1]:base[1]+H] = 4
        self.W, self.H = self.global_resolution

        # [Get Color Map]
        root = config.root
        name = "northrend"
        self.color_map = load_json(root + "/env/mkgrid/domains/{}_color_map.json".format(name))
        self.int2colors = np.array([self.color_map[c] for c in self.color_map])

        # [Agent Position]
        self.agent_x = H//2
        self.agent_y = W//2
        self.agent_dir = torch.tensor([1,0]).int()

    def reset(self):
        return 
    
        
    def render(self):
        global_frame = self.int2colors[self.grid]
        
        base = self.base
        locW, locH = self.local_resolution
        cx, cy = base[0]+int(self.agent_x), base[1]+int(self.agent_y)
        global_frame[cx][cy] = torch.tensor([1,1,0])
        local_frame = global_frame[
            cx-locW//2:cx+locW//2,
            cy-locH//2:cy+locH//2
            ]
        
        return local_frame, global_frame

    def step(self, action, state = None):
        if state is None:
            next_state = self.grid
        else: next_state = state
        if action == 0:
            self.agent_x += self.agent_dir[0]
            self.agent_y += self.agent_dir[1]
        if action == 1:
            self.agent_dir[0] += 1
        if action == 2:
            self.agent_dir[1] += 1
        if action == 3:
            self.agent_dir[0] -= 1
        if action == 4:
            self.agent_dir[1] -= 1
        if action == 5:
            pass
        if self.agent_x < 0: self.agent_x = 0
        if self.agent_x > self.W-1: self.agent_x = self.W-1
        if self.agent_y < 0: self.agent_y = 0
        if self.agent_y > self.H-1: self.agent_y = self.H-1
        done = 0
        outputs= {"prev_state":state,
                  "next_state":next_state,
                  "done":done,
                  "reward":1.0}
        return outputs

if __name__ == "__main__":
    import json
    def save_json(data,path):
        '''input the diction data and save it'''
        beta_file = json.dumps(data)
        file = open(path,'w')
        file.write(beta_file)
        return True

    def load_json(path):
        with open(path,'r') as f:
            data = json.load(f)
            return data


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
    def hex_to_rgb(value):
        value = value.lstrip('#')
        lv = len(value)
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    
    color_map_dict = {}
    for i,c in enumerate(northrend_colors):color_map_dict[str(i)] = hex_to_rgb(c)
    save_json(color_map_dict, "color_map.json")