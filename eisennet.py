from colorsys import rgb_to_hls
import warnings
warnings.filterwarnings("ignore")

from datasets import *
from models import *
from config import *

sidelength = 128
object3d_dataset = Objects3dDataset(config, sidelength, depth_aug = True, multiview_aug= True, stage = 1)
dataloader = DataLoader(object3d_dataset, batch_size = 1, shuffle = True)

for sample in dataloader:
    model_input, gt = sample
    break



for name in model_input:
    print(name, model_input[name].shape)

# show some 3d-point cloud data.
from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D




# set up the limit of x,y,z and normalize them
print(model_input["cam_poses"])


def visualize(input_pcs):
    rang = 0.4; N = len(input_pcs)
    fig = plt.figure(figsize=plt.figaspect(1/N), frameon = True)
    
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

point_cloud = model_input["point_cloud"]
rgb = model_input["rgb"]
coords = model_input["coords"]
occ = model_input["occ"]
coord_color = model_input["coord_color"]



input_pcs = [(coords[0,:,:] * (occ[0,:,:]+1)/ 2,coord_color[0,:,:]),
             (point_cloud[0,:,:],rgb[0,:,:]),
             (coords[0,:,:] * (occ[0,:,:]+1)/ 2,coord_color[0,:,:])]

visualize(input_pcs)
plt.show()

"""
ax.scatter(point_cloud[b,:,0],point_cloud[b,:,1],point_cloud[b,:,2], c = rgb[b,:,:] / 255)
B = 1
test_dataset = SpriteData(split = "train")
#test_dataset = ToyData(split = "train")
test_dataset = AcherusDataset(split = "train")
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = B, shuffle = True)

for sample in dataloader:ims = sample["image"];break;

esnet = EisenNet(config)

outputs = esnet(ims)

masks = outputs["masks"]
B,W,D,N = masks.shape

plt.figure("segments")
for i in range(N):
    for b in range(B):
        plt.subplot(B,N + 1,1 + i + b * (N+1))
        plt.imshow(masks[b,:,:,i], cmap = "bone")
    plt.subplot(B,N+1,(b+1) * (N+1))
    plt.imshow(ims[b,:,:,:])
plt.show()
"""