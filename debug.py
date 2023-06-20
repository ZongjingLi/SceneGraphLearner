from models.percept.construct_net import *
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

construct_net = ConstructNet(config)

outputs = construct_net(ims)

level_masks = outputs["masks"]

for level in level_masks:
    print(level.shape)

base_mask = level_masks[0].reshape([10,128,128]).detach()
print(base_mask.max())

for i in range(base_mask.shape[0]):
    plt.subplot(1, 2, 1);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False);
    plt.imshow(base_mask[i], cmap="bone")
    plt.subplot(1, 2, 2);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False);

    plt.imshow(base_mask[i].unsqueeze(-1) * ims[0])
    plt.pause(0.01)
plt.show()

B = 1

for b in range(B):
    plt.subplot(1, B, b + 1);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False); plt.imshow(ims[b,:,:,:])
plt.show()