from colorsys import rgb_to_hls
import warnings
warnings.filterwarnings("ignore")

from datasets import *
from models import *
from config import *

sidelength = 128
object3d_dataset = Objects3dDataset(config, sidelength, depth_aug = True, multiview_aug= True)
dataloader = DataLoader(object3d_dataset, batch_size = 1, shuffle = True)

for sample in dataloader:
    model_input, gt = sample
    break

point_cloud = model_input["point_cloud"]
rgb = model_input["rgb"]
coords = model_input["coords"]
occ = model_input["occ"]
coord_color = model_input["coord_color"]

for name in model_input:
    print(name, model_input[name].shape)

# show some 3d-point cloud data.
from types import new_class
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = Axes3D(fig)

rang = 0.5
# set up the limit of x,y,z and normalize them
ax.set_zlim(-rang,rang)
ax.set_xlim(-rang,rang)
ax.set_ylim(-rang,rang)

print(model_input["cam_poses"])
b = 0
coords = coords *  (occ[b,:,:]+1)/ 2
ax.scatter(coords[b,:,0],coords[b,:,1],coords[b,:,2], c = coord_color[b,:,:] )
plt.show()


ax.scatter(point_cloud[b,:,0],point_cloud[b,:,1],point_cloud[b,:,2], c = rgb[b,:,:] / 255)
plt.show()



"""
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