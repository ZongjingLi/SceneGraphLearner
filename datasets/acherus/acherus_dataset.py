'''
 # @ Author: Yiqi Sun
 # @ Create Time: 2023-03-14 13:31:26
 # @ Modified by: Yiqi Sun
 # @ Modified time: 2023-03-14 13:32:10
 # @ Description: This file is distributed under the MIT license.
'''
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
from utils import *

class MineClip(Dataset):
    def __init__(self,split = "train",path = "",resolution = (480,360)):
        super().__init__()
        self.resolution = resolution
        self.path = "data/minevs/{}.jpg"
        #self.images = sorted(glob(self.path))
        self.img_transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def __len__(self):
        return 10

    def __getitem__(self, index):
        image = Image.open(self.path.format(index))
        image = image.convert("RGB").resize(self.resolution)
        image = self.img_transform(image)
        sample = {"image":image}
        return sample