import torch
import argparse 

from models import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_root",       default = "/Users/melkor/Documents/datasets")
parser.add_argument("--device",             default = device)
parser.add_argument("--name",               default = "SceneGraphLearner")
parser.add_argument("--domain",             default = "toy")

# setup the perception module
parser.add_argument("--perception",         default = "valkyr")
parser.add_argument("--perception_size",    default = 7)
parser.add_argument("--imsize",             default = 128)
parser.add_argument("--resolution",         default = (128,128))
parser.add_argument("--hidden_dim",         default = 100)
parser.add_argument("--latent_dim",         default = 128) # point cloud encoder
parser.add_argument("--scaling",            default = 10.0)
parser.add_argument("--acne_net_depth",     default = 3)
parser.add_argument("--acne_num_g",         default = 10)
parser.add_argument("--acne_dim",           default = 128)
parser.add_argument("--acne_bn_type",       default = "bn")
parser.add_argument("--cn_type", type=str,
                      default="acn_b",
                      help="Encoder context normalization type")
parser.add_argument("--node_feat_dim",      default = 100)

parser.add_argument("--object_num",         default = 7)
parser.add_argument("--part_num",           default = 3)

# setup the concept learner 
parser.add_argument("--concept_projection", default = True)
parser.add_argument("--concept_type",       default = "cone")
parser.add_argument("--concept_dim",        default = 100)
parser.add_argument("--object_dim",         default = 100)
parser.add_argument("--temperature",        default = 0.5)

# box concept methods
parser.add_argument("--method",             default = "uniform")
parser.add_argument("--offset",             default = [-.25, .25])
parser.add_argument("--center",             default =[.0, .0])
parser.add_argument("--entries",            default = 10)
parser.add_argument("--translator",         default = translator)


# hiearchy graph generation
parser.add_argument("--global_feature_dim", default = 66)

# intuitive physics model and particle filter

config = parser.parse_args(args = [])

def str2bool(x):return x.lower in ("true","1")