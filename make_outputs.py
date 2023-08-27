from visualize.visualize_pointcloud import *
import numpy as np
import torch
import argparse

visparser = argparse.ArgumentParser()
visparser.add_argument("--model",           default = "csq_net")
visconfig = visparser.parse_args()

def hex2rgb(h):
    return tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb2hex(rgb):
    return '#{0:02x}{1:02x}{2:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

colors = torch.tensor(np.array([[float(c)/255.0 for c in hex2rgb(color)] for color in colors]))

if visconfig.model in ["csq_net"]:
    """
        point_cloud_masks = []
    masks = torch.tensor(np.load("outputs/masks.npy")) 
    num_masks = masks.shape[0]
    for i in range(num_masks):
        print(masks[i].max(),masks[i].min())
        occ_color = torch.tensor(1-masks[i]).unsqueeze(-1) * (colors[i % len(colors)].unsqueeze(0).repeat(n,1))
        point_cloud_masks.append([pc ,occ_color])
    visualize_pointcloud(point_cloud_masks)
    plt.show()
    """
    coords = np.load("outputs/recon_point_cloud.npy")
    n = coords.shape[0]
    coords_colors = torch.ones([n,3]) * 0.5

    pc = torch.tensor(np.load("outputs/point_cloud.npy"))
    pc_colors = torch.ones(pc.shape[0],3) * 0.5

    input_pcs = [(coords,coords_colors),(pc,pc_colors)]

    visualize_pointcloud(input_pcs)
    plt.show()
    
    masks = torch.tensor(np.load("outputs/masks.npy")) 
    #print(masks.sum(1))
    #print(masks.max(dim = 1).values)
    vis_pts(pc.permute(1,0).unsqueeze(0),masks.permute(1,0).unsqueeze(0))
    plt.show()



if visconfig.model in ["point_net"]:
    point_cloud = np.load("outputs/point_cloud.npy")
    rgb = np.load("outputs/rgb.npy")
    coords = np.load("outputs/coords.npy")
    occ = np.load("outputs/occ.npy")
    coord_color = np.load("outputs/coord_color.npy")

    recon_occ = np.load("outputs/recon_occ.npy")
    recon_coord_color = np.load("outputs/recon_coord_color.npy")

    recon_probs = recon_occ
    #recon_probs = (occ+1)/2

    input_pcs = [(coords * (occ+1)/ 2,coord_color),
             (point_cloud,rgb),
             (coords * recon_probs,recon_coord_color)]

    visualize_pointcloud(input_pcs)
    plt.show()