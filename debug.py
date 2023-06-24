from models.percept.construct_net import *
from datasets import *
from config import *

# [Grid-Line Domain Diff]
def lin2img(lin,b):return lin.reshape([b,128,128,3])

test_dataset = SpriteData(split = "train")
test_dataset = ToyData(split = "train")
test_dataset = AcherusDataset(split = "train")
dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)

for sample in dataloader:
    ims = sample["image"]
    break;

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            
            for r in range(10):
                random_long_range = torch.randint(128, (1,2) )[0]
                edges[0].append(random_long_range[0] // size)
                edges[1].append(random_long_range[1] % size)
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        if 1 and (i+dx) * size + (j + dy) != loc:
                            edges[0].append(loc)
                            edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

construct_net = ConstructNet(config)

outputs = construct_net(ims)


level_masks = outputs["masks"]
level_index = outputs["level_index"]
level_inds = outputs["level_index"]
level_features = outputs["level_features"]

for level in level_masks:print(level.shape)

for idx in level_index:print(idx)

for feat in level_features:print(feat.shape)


base_locs = level_index[0]
base_mask = level_masks[0].reshape([10,128,128]).detach()
print(base_mask.max())

for i in range(base_mask.shape[0]):
    plt.subplot(1, 2, 1);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False);
    plt.imshow(base_mask[i], cmap="bone")
    plt.subplot(1, 2, 2);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False);

    plt.imshow(base_mask[i].unsqueeze(-1) * ims[0])
    bx = base_locs[i] / 128
    by = base_locs[i] % 128
    plt.scatter(bx,by)
    plt.pause(0.5)
plt.show()

B = 1

for b in range(B):
    plt.subplot(1, B, b + 1);plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False); plt.imshow(ims[b,:,:,:])
    plt.pause(0.5)
plt.show()