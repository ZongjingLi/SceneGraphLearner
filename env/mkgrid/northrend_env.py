import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

class Northrend(nn.Module):
    def __init__(self, config):
        super().__init__()
        H, W = config.env_global_resolution    
        locH, locW = config.env_local_resolution

        global_map = None

        self.global_resolution = (H,W)
        self.local_resolution = (locH,locW)
        # [Build Global Grid]
        self.grid = np.zeros(H,W)

        # [Get Color Map]
        root = config.root
        name = "northrend"
        self.color_map = load_json(root + "/env/mkgrid/domains/{}_color_map.json".format(name))
        self.int2colors = np.array([c in self.color_map])

        self.agent_x = W//2
        self.agent_y = H//2
        
    def render(self):
        global_frame = self.int2colors[self.grid]
        cx, cy = self.agent_x, self.agent_y
        local_frame = global_frame[cx, cy]
        return local_frame, global_frame

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