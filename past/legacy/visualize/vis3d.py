# show some 3d-point cloud data.
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import torch.nn as nn

fig = plt.figure()
ax = Axes3D(fig)

N = 100
x_points = torch.randn([N, 3]) * 0.5 + torch.tensor([-0.,-0,5]).unsqueeze(0).repeat([N,1])
y_points = torch.randn([N, 3]) * 0.3 + torch.tensor([0.2,2,2]).unsqueeze(0).repeat([N,1])

scale = 1
x = torch.linspace(-1, 1, 100) * scale
y = torch.linspace(-1, 1, 100) * scale
X,Y = torch.meshgrid(x,y)
Z = X**2 + Y**2 

#ax.plot_surface(X,Y,Z, cmap = "rainbow")

ax.plot_surface(X,Y,Z * 0, cmap = "winter")

delta_x = 0.02
delta_y = 0.01
size = 0.03

for i in range(100):
    plt.cla()
    ax.set_zlim(0.0,1.0)
    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(-1.0,1.0)
    x_points = torch.randn([N, 3]) * size + torch.tensor([-1. + delta_x * i,-1. + delta_y * i,0.3]).unsqueeze(0).repeat([N,1])
    y_points = torch.randn([N, 3]) * size + torch.tensor([1. - delta_x * i,-1. + delta_y * i,0.3]).unsqueeze(0).repeat([N,1])
    ax.plot_surface(X,Y,Z * 0, color = "grey", alpha = 0.1)
    ax.scatter(x_points[:,0], x_points[:,1], x_points[:,2], color = "red")
    ax.scatter(y_points[:,0], y_points[:,1], y_points[:,2], color = "cyan")
    plt.pause(0.01)
#ax.scatter(y_points[:,0], y_points[:,1], y_points[:,2], color = "cyan")

plt.show()
