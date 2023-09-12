import torch
import argparse 

from models import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count,
              "parents":Parents,"subtree":Subtree}

LOCAL = True

root_path = "/Users/melkor/Documents/GitHub/SceneGraphLearner" if LOCAL else "SceneGraphLearner"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("--root",                   default = root_path)
parser.add_argument("--dataset_root",           default = "/Users/melkor/Documents/datasets")
parser.add_argument("--device",                 default = device)
parser.add_argument("--name",                   default = "SceneGraphLearner")

parser.add_argument("--domain",                 default = "toy")
parser.add_argument("--category",               default = "")

#parser.add_argument("--domain",             default = "structure")
#parser.add_argument("--category",           default = ["vase"])

# setup the perception module
parser.add_argument("--perception",             default = "valkyr")
parser.add_argument("--channel",                default = 3)
parser.add_argument("--spatial_dim",            default = 2)
parser.add_argument("--fourier_dim",            default = 7)
parser.add_argument("--perception_size",        default = 3)
parser.add_argument("--imsize",                 default = 128)
parser.add_argument("--resolution",             default = (128,128))
parser.add_argument("--conv_feature_dim",       default = 32)
parser.add_argument("--hidden_dim",             default = 100)
parser.add_argument("--latent_dim",             default = 128) # point cloud encoder
parser.add_argument("--scaling",                default = 10.0)

# acne network
parser.add_argument("--num_pts",                default = 1000)
parser.add_argument("--indim",                  default = 3)
parser.add_argument("--grid_dim",               default = 64)
parser.add_argument("--decoder_grid",           default = "learnable")
parser.add_argument("--decoder_bottleneck_size",    default = 1280) # 1280
parser.add_argument("--acne_net_depth",         default = 3)
parser.add_argument("--acne_num_g",             default = 10)
parser.add_argument("--acne_dim",               default = 128)
parser.add_argument("--acne_bn_type",           default = "bn")
parser.add_argument("--cn_type", type=str,
                      default="acn_b",
                      help="Encoder context normalization type")
parser.add_argument("--node_feat_dim",          default = 100)
parser.add_argument("--pose_code",              default = "nl-noR_T")

# concept learner structure
parser.add_argument("--object_num",             default = 11)
parser.add_argument("--part_num",               default = 3)
parser.add_argument("--hierarchy_latent",       default = 128)
parser.add_argument("--hierarchy_construct",    default = [3])

# setup the concept learner 
parser.add_argument("--concept_projection",     default = True)
parser.add_argument("--concept_type",           default = "cone")
parser.add_argument("--concept_dim",            default = 100)
parser.add_argument("--object_dim",             default = 100)
parser.add_argument("--temperature",            default = 32. * 4)

# box concept methods
parser.add_argument("--method",                 default = "uniform")
parser.add_argument("--offset",                 default = [-.25, .25])
parser.add_argument("--center",                 default =[.0, .0])
parser.add_argument("--entries",                default = 32 * 3)
parser.add_argument("--translator",             default = translator)


# hiearchy graph generation
parser.add_argument("--global_feature_dim", default = 66)

# [Physics] intuitive physics model and particle filter
parser.add_argument("--position_dim",           default = 3)
parser.add_argument("--state_dim",              default = 6)
parser.add_argument("--attr_dim",               default = 4)
parser.add_argument("--relation_dim",           default = 1)
parser.add_argument("--particle_feature_dim",   default = 128)
parser.add_argument("--relation_feature_dim",   default = 100)
parser.add_argument("--prop_feature_dim",       default = 132)
parser.add_argument("--history_window",         default = 5)
parser.add_argument("--roll_outs",              default = 5)
parser.add_argument("--action_dim",             default = 0)
parser.add_argument("--observation",            default = "full")

# [Neuro Particle Filter]
parser.add_argument("--physics_model",          default = "agtnet")
parser.add_argument("--type_penalty",           default = 1.)
parser.add_argument("--distance_penalty",       default = 1.)
parser.add_argument("--color_penalty",          default = 1.)
parser.add_argument("--distance_threshold",     default = 0.3)
parser.add_argument("--base_penalty",           default = 1.)

# [Environment Config]
parser.add_argument("--env_name",               default = "MKGrid")
parser.add_argument("--global_env_resolution",  default = (64,64))
parser.add_argument("--local_env_resolution",   default = (32,32))

config = parser.parse_args(args = [])

def str2bool(x):return x.lower in ("true","1")