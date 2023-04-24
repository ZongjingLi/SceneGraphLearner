import torch
import torch.nn as nn

from config import *
from models import *
from datasets import *

model = SceneLearner(config)
model.scene_perception = torch.load("checkpoints/PTRObjects_toy_slot_attention.ckpt", map_location=config.device)

train_dataset = PTRData("train", resolution = config.resolution)
dataloader = DataLoader(train_dataset, batch_size = 2)

counter = 0
n = 4

import matplotlib.pyplot as plt

for sample in dataloader:
    ims = sample["image"]
    image = ims.permute([0,3,1,2])
    b,c,w,h = image.shape
    # encoder model: extract visual feature map from the image

    feature_map = model.scene_perception.encoder_net(image)
    recons = model.scene_perception(ims)["full_recons"]
    
    for i in range(feature_map.shape[1]):
        grad_map = feature_map[0,i,:,:]
        plt.subplot(131)
        plt.imshow(grad_map.detach().numpy(),cmap="bone")
        plt.subplot(132)
        plt.imshow(ims[0,:,:,:])
        plt.subplot(133)
        plt.imshow(recons[0,:,:,:].detach().numpy())
        plt.show()

    counter += 1
    if counter >= n:break


    