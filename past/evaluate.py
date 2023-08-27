import torch
import torch.nn as nn
from tqdm import tqdm

from models import *
from datasets import *
from config import *

def evaluate_pointcloud(train_model, config, args, phase = "1"):
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = True if args.training_mode in ["joint", "query"] else False
    print("\nstart the evaluation: {} query:[{}]".format(args.name,query))
    print("evaluation config: \nbatch: {} samples \n".format(args.batch_size))

    #[setup the training and validation dataset]
    if args.dataset == "Objects3d":
        train_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))

    dataloader = DataLoader(train_dataset, batch_size = int(args.batch_size), shuffle = args.shuffle)

    # [setup the optimizer and lr schedulr]

    # [start the training process recording]
    itrs = 0

    for epoch in range(1):
        epoch_loss = 0
        for sample in tqdm(dataloader):
            sample, gt = sample
            # [perception module training]
            point_cloud = sample["point_cloud"]
            rgb = sample["rgb"]
            coords = sample["coords"]
            occ = sample["occ"]
            coord_color = sample["coord_color"]

            outputs = model.scene_perception(sample)
            all_losses = {}

            # [Perception Loss]
            perception_loss = 0
            for loss_name in outputs["loss"]:
                weight = args.loss_weights[loss_name]
                perception_loss += outputs["loss"][loss_name] * weight

            recon_occ = outputs["occ"]
            recon_coord_color = outputs["color"]

            # [Language Loss]
            language_loss = 0
  
            # [calculate the working loss]
            working_loss = perception_loss * 1 + language_loss * 1
            try:epoch_loss += working_loss.detach().numpy()
            except:epoch_loss += working_loss

            if not(itrs % 10):      
                np.save("outputs/point_cloud.npy",np.array(point_cloud[0,:,:]))
                np.save("outputs/rgb.npy",np.array(rgb[0,:,:]))
                np.save("outputs/coords.npy",np.array(coords[0,:,:]))
                np.save("outputs/occ.npy",np.array(occ[0,:,:]))
                np.save("outputs/coord_color.npy",np.array(coord_color[0,:,:]))

                np.save("outputs/recon_occ.npy",np.array(recon_occ[0,:].unsqueeze(-1).detach()))
                np.save("outputs/recon_coord_color.npy",np.array(recon_coord_color[0,:,:].detach()))
            itrs += 1

    print("\n\nExperiment {} : Training Completed.".format(args.name))
    print("Epoch Loss: {}".format(epoch_loss))

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0}

evalparser = argparse.ArgumentParser()
evalparser.add_argument("--name",               default = "WLK")
evalparser.add_argument("--device",             default = config.device)
evalparser.add_argument("--dataset",            default = "Objects3d")
evalparser.add_argument("--batch_size",         default = 2)
evalparser.add_argument("--phase",              default = 1)
evalparser.add_argument("--shuffle",            default = True)
evalparser.add_argument("--perception",         default = "point_net")
evalparser.add_argument("--training_mode",      default = "3d_perception")
evalparser.add_argument("--loss_weights",       default = weights)
evalparser.add_argument("--checkpoint_dir",     default = "checkpoints/scenelearner/3dpc/KFT_3d_perception_toy_point_net_phase1.pth")

evalargs = evalparser.parse_args()
config.perception = evalargs.perception

if evalargs.checkpoint_dir:
    #model = torch.load(args.checkpoint_dir, map_location = config.device)
    model = SceneLearner(config)
    model.load_state_dict(torch.load(evalargs.checkpoint_dir, map_location=evalargs.device))
else:
    print("No checkpoint to load and creating a new model instance")
    model = SceneLearner(config)
model = model.to(evalargs.device)

print("start evaluation:")

if evalargs.dataset in ["Objects3d"]:
    print("start the 3d point cloud model training.")
    evaluate_pointcloud(model, config, evalargs, phase = evalargs.phase)

if evalargs.dataset in ["Sprites","Acherus","Toys","PTR"]:
    print("start the image domain training session.")
    #train_pointcloud(model, config, evalargs)

