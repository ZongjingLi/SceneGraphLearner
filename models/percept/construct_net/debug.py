
from .construct_net import *
from datasets import *
from config import *
# [Grid-Line Domain Diff]
def lin2img(lin,b):return lin.reshape([b,128,128,3])

test_dataset = SpriteData(split = "train")
#test_dataset = ToyData(split = "train")
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True)

for sample in dataloader:
    ims = sample["image"]
    break;