from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def visualize_pointcloud(fig, input_pcs,name="pc"):
    rang = 0.4; N = len(input_pcs)

    for i in range(N):
        ax = fig.add_subplot(1, N, 1 + i, projection='3d')
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
        ax.view_init(elev = -80, azim = -90)
        coords = input_pcs[i][0]
        colors = input_pcs[i][1]
        ax.scatter(coords[:,0],coords[:,1],coords[:,2], c = colors)
    plt.savefig("outputs/{}.png".format(name))