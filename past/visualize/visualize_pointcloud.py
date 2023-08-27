from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import torch

def vis_pts(x, att=None, vis_fn="outputs/temp.png"):
    idx = 0
    pts = x[idx].transpose(1, 0).cpu().numpy()
    label_map = None
    if att is not None:
        label_map = torch.argmax(att, dim=2)[idx].squeeze().cpu().numpy()
    vis_pts_att(pts, label_map, fn=vis_fn)

def visualize_attention_masks(points, attn):
    B,K,N = attn.shape # [BxKxN]
    plt.figure("Attention Masks")
    for i in range(K):
        pass

def visualize_pointcloud_components(pts,attn):
    B,K,N = attn.shape
    fig = plt.figure("Point Cloud Components")
    for k in range(K):
        ax = fig.add_subplot(1, N , 1 + i, projection='3d')

def vis_pts_att(pts, label_map, fn="outputs/temp.png", marker=".", alpha=0.9):
    # pts (n, d): numpy, d-dim point cloud
    # label_map (n, ): numpy or None
    # fn: filename of visualization
    assert pts.shape[1] in [2, 3]
    if pts.shape[1] == 2:
        xs = pts[:, 0]
        ys = pts[:, 1]
        if label_map is not None:
            plt.scatter(xs, ys, c=label_map, cmap="jet", marker=".", alpha=0.9, edgecolor="none")
        else:
            plt.scatter(xs, ys, c="grey", alpha=0.8, edgecolor="none")
        # save
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.axis("off")
    elif pts.shape[1] == 3:
        TH = 0.7
        fig_masks = plt.figure("part visualization")
        ax_masks = fig_masks.add_subplot(111, projection="3d")
        ax_masks.set_zlim(-TH,TH)
        ax_masks.set_xlim(-TH,TH)
        ax_masks.set_ylim(-TH,TH)
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        if label_map is not None:
            ax_masks.scatter(xs, ys, zs, c=label_map, cmap="jet", marker=marker, alpha=alpha)
        else:
            ax_masks.scatter(xs, ys, zs, marker=marker, alpha=alpha, edgecolor="none")
        ax_masks.view_init(elev = 100, azim = -120)
        ax_masks.set_xticklabels([])
        ax_masks.set_yticklabels([])
        ax_masks.set_zticklabels([])


        for axis in [ax_masks.xaxis, ax_masks.yaxis, ax_masks.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax_masks.set_axis_off()

    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    

def visualize_pointcloud(input_pcs,name="pc"):
    rang = 1.0; N = len(input_pcs)
    num_rows = 3
    fig = plt.figure("visualize",figsize=plt.figaspect(1/N), frameon = True)
    for i in range(N):
        ax = fig.add_subplot(1, N , 1 + i, projection='3d')
        ax.set_zlim(-rang,rang);ax.set_xlim(-rang,rang);ax.set_ylim(-rang,rang)
        # make the panes transparent
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.set_ticklabels([])
            axis._axinfo['axisline']['linewidth'] = 1
            axis._axinfo['axisline']['color'] = (0, 0, 0)
            axis._axinfo['grid']['linewidth'] = 0.5
            axis._axinfo['grid']['linestyle'] = "-"
            axis._axinfo["grid"]['color'] =  (1,1,1,0)
            axis._axinfo['tick']['inward_factor'] = 0.0
            axis._axinfo['tick']['outward_factor'] = 0.0
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.set_axis_off()
        #ax.view_init(elev = -80, azim = -90)
        coords = input_pcs[i][0]
        colors = input_pcs[i][1]
        ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors)
    plt.savefig("outputs/{}.png".format(name))
