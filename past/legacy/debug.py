from config     import *

from legacy import freeze_parameters
from models     import *
from datasets   import *

import networkx as nx

from visualize.visualize_pointcloud import vis_pts
from visualize.visualize_pointcloud import visualize_pointcloud
from datasets.p3d_dataset.structure_net.generate_structure_qa import dfs_point_cloud, get_leafs

# [Create a Dataset]
B = 1
dataset = StructureGroundingDataset(config, category="chair", split = "train")
dataloader = DataLoader(dataset, batch_size = B, shuffle = True)

# [Get A Sample Data]
for sample in dataloader:
    sample, gt = sample

EPS = 1e-5
config.perception = "csqnet"
config.training_mode = "3d_perception"
config.concept_type = "box"
config.concept_dim = 100
config.domain = "structure"

def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]


learner = SceneLearner(config)
config.hierarchy_construct = (7,5,3)
#learner.load_state_dict(torch.load("checkpoints/VNL_3d_perception_structure_csqnet_phase0.pth",map_location = "cpu"))
optimizer = torch.optim.RMSprop(learner.parameters(), lr = 1e-4)

print(sample["programs"])

print(sample["questions"])

print(sample["answers"])


# plot the point cloud structure
outputs = learner.part_perception(sample)
point_cloud = sample["point_cloud"][0]
masks = outputs["masks"][0]
category = sample["category"]
index = sample["index"]


vis_pts(point_cloud.permute(1,0).unsqueeze(0),masks.permute(1,0).unsqueeze(0))

# Color the Ground Truth
hier_path = root + "/partnethiergeo/{}_hier/{}.json".format(category[0], index[0])
hier_data = load_json(hier_path)
pc_path = root + "/partnethiergeo/{}_geo/{}.npz".format(category[0], index[0])
part_pts = np.load(pc_path)
pts, rgbs = dfs_point_cloud(part_pts["parts"], get_leafs(hier_data))

# visualize point clouds and ground truth
visualize_pointcloud([
    (point_cloud, torch.ones([1000,1])),
    (pts, rgbs)
])

# plot the scene tree level
scene_tree_path = sample["scene_tree"][-1]
scene_tree = nx.read_gpickle(scene_tree_path)
plt.figure("scene tree")
nx.draw_networkx(scene_tree)

plt.show()

freeze_parameters(learner.part_perception)

visualize = True
for epoch in range(10000):
    g = -0.5; r = 0.3
    outputs = learner.part_perception(sample)

    features,masks,positions = outputs["features"],outputs["masks"],outputs["positions"] 

    if visualize and epoch == 0:
        recon_pc = outputs["recon_pc"][0]
        point_cloud = sample["point_cloud"][0]
        masks = outputs["masks"][0]
        np.save("outputs/recon_point_cloud.npy",np.array(recon_pc.cpu().detach()))
        np.save("outputs/point_cloud.npy",np.array(point_cloud.cpu().detach()))
        np.save("outputs/masks.npy",np.array(masks.cpu().detach()))


    features = learner.feature2concept(features)
    #features = torch.cat([features, torch.ones([1,features.shape[1],config.concept_dim]) * EPS], dim = -1)

    qa_programs = sample["programs"]
    answers = sample["answers"]
    scene = learner.build_scene(features)
    for b in range(B):
        scores,features,connections = load_scene(scene, b)

        kwargs = {"features":features,
                  "end":scores,
                 "connections":connections}

        language_loss = 0
        for i,q in enumerate(qa_programs):
            q = learner.executor.parse("subtree(scene())")

            o = learner.executor(q, **kwargs)
            o["end"].reverse()

            #for s in o["end"]:print(np.array((torch.sigmoid(s) + 0.5).int()))

            #q = learner.executor.parse("exist(filter(subtree(scene()),chair ))")
            q = learner.executor.parse("exist(filter( subtree(scene()) ,body) )")

            o = learner.executor(q, **kwargs)
            #print(o["end"])
            language_loss += (1 + torch.sigmoid( (o["end"] + g) )/r)
        
        optimizer.zero_grad()
        language_loss.backward()
        optimizer.step()

        
        o = learner.executor(q, **kwargs)
        sys.stdout.write("\r p:{}".format(float(torch.sigmoid( (o["end"] + g)/r ).detach()) ) )