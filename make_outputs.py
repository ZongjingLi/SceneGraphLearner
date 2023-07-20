from visualize.visualize_pointcloud import *
import numpy as np

point_cloud = np.load("outputs/point_cloud.npy")
rgb = np.load("outputs/rgb.npy")
coords = np.load("outputs/coords.npy")
occ = np.load("outputs/occ.npy")
coord_color = np.load("outputs/coord_color.npy")

recon_occ = np.load("outputs/recon_occ.npy")
recon_coord_color = np.load("outputs/recon_coord_color.npy")

input_pcs = [(coords * (occ+1)/ 2,coord_color),
             (point_cloud,rgb),
             (coords * (recon_occ+1)/2,recon_coord_color)]

visualize_pointcloud(input_pcs)
plt.show()