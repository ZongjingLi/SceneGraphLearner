import warnings
warnings.filterwarnings("ignore")

import tensorflow

import torch
import argparse 
import datetime
import time
import sys

from datasets import *

from config import *
from models import *
from visualize.answer_distribution import *
from visualize.visualize_pointcloud import *

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color

from train import *


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_hierarchy(model, depth):
    size = len(model.scene_builder)
    for i in range(1,size+1):
        if i <= depth:unfreeze_parameters(model.hierarhcy_maps[i-1])
        else:freeze_parameters(model.scene_builder)
    model.executor.effective_level = depth


def load_scene(scene, k): 
    scores = scene["scores"]; features = scene["features"]; connections = scene["connections"]
    return [score[k] for score in scores], [feature[k] for feature in features], \
        [connection[k] for connection in connections[1:]]

weights = {"reconstruction":1.0,"color_reconstruction":1.0,"occ_reconstruction":1.0,"localization":1.0,"chamfer":1.0,"equillibrium_loss":1.0}

argparser = argparse.ArgumentParser()

argparser.add_argument("--mode",                    default = "scenelearner")

# [general config of the training]
argparser.add_argument("--phase",                   default = "0")
argparser.add_argument("--device",                  default = config.device)
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 400 * 3)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 1e-3)
argparser.add_argument("--batch_size",              default = 1)
argparser.add_argument("--dataset",                 default = "toy")
argparser.add_argument("--category",                default = ["vase"])
argparser.add_argument("--freeze_perception",       default = False)
argparser.add_argument("--concept_type",            default = False)

# [perception and language grounding training]
argparser.add_argument("--perception",              default = "psgnet")
argparser.add_argument("--training_mode",           default = "joint")
argparser.add_argument("--alpha",                   default = 1.00)
argparser.add_argument("--beta",                    default = 1.0)
argparser.add_argument("--loss_weights",            default = weights)

# [additional training details]
argparser.add_argument("--warmup",                  default = True)
argparser.add_argument("--warmup_steps",            default = 300)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [curriculum training details]
argparser.add_argument("--effective_level",         default = 1)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 10,       type=int)
argparser.add_argument("--pretrain_perception",     default = False)
argparser.add_argument("--visualize_batch",         default = 2)

# [reinforcnment learning setup]
argparser.add_argument("--env_name",                default = "None")
argparser.add_argument("--traj_sample_num",         default = 2)

args = argparser.parse_args()

config.perception = args.perception
if args.concept_type: config.concept_type = args.concept_type
args.freeze_perception = bool(args.freeze_perception)
args.lr = float(args.lr)
args.batch_size = int(args.batch_size)


if args.checkpoint_dir:
    #model = torch.load(args.checkpoint_dir, map_location = config.device)
    model = AutoLearner(config)
    if "ckpt" in args.checkpoint_dir[-4:]:
        model.scenelearner = torch.load(args.checkpoint_dir, map_location = args.device)
    else: model.load_state_dict(torch.load(args.checkpoint_dir, map_location=args.device))
else:
    print("No checkpoint to load and creating a new model instance")
    model = AutoLearner(config)
model = model.to(args.device)


if args.pretrain_perception:
    model.load_state_dict(torch.load(args.pretrain_perception, map_location = config.device))


print("using perception: {} knowledge:{} dataset:{}".format(args.perception,config.concept_type,args.dataset))

if args.mode == "planning":
    train_rl(model.planning_model, config, args)

if args.mode == "physics":
    train_physics(model, config, args)

if args.mode == "scenelearner":
    if args.dataset in ["Objects3d","StructureNet"]:
        print("start the 3d point cloud model training.")
        train_pointcloud(model.scenelearner, config, args, phase = args.phase)

    if args.dataset in ["Sprites","Acherus","Toys","PTR"]:
        print("start the image domain training session.")

        if args.name in ["VKY"]:
            train_scenelearner(model.scenelearner, config, args)
        else:
            train_image(model.scenelearner, config, args)



