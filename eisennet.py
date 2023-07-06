import warnings
warnings.filterwarnings("ignore")

from datasets import *
from models import *
from config import *


B = 1

test_dataset = SpriteData(split = "train")
#test_dataset = ToyData(split = "train")
test_dataset = AcherusDataset(split = "train")
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = B, shuffle = True)

for sample in dataloader:ims = sample["image"];break;

esnet = EisenNet(config)

outputs = esnet(ims)

B,W,D,N = outputs.shape
print(outputs.shape)

plt.figure("segments")
for i in range(N):
    for b in range(B):
        plt.subplot(B,N + 1,1 + i + b * (N+1))
        plt.imshow(outputs[b,:,:,i], cmap = "bone")
    plt.subplot(B,N+1,(b+1) * (N+1))
    plt.imshow(ims[b,:,:,:])
plt.show()