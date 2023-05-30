import torch
import argparse 

from models import *
translator = {"scene":Scene,"exist":Exist,"filter":Filter,"union":Union,"unique":Unique,"count":Count}

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--device",             default = device)
parser.add_argument("--name",               default = "SceneGraphLearner")
parser.add_argument("--domain",             default = "toy")

# setup the perception module
parser.add_argument("--perception",         default = "slot_attention")
parser.add_argument("--perception_size",    default = 2)
parser.add_argument("--imsize",             default = 128)
parser.add_argument("--resolution",         default = (128,128))
parser.add_argument("--hidden_dim",         default = 100)

parser.add_argument("--object_num",         default = 5)
parser.add_argument("--part_num",           default = 3)

# setup the concept learner 
parser.add_argument("--concept_type",       default = "cone")
parser.add_argument("--concept_dim",        default = 102)
parser.add_argument("--object_dim",         default = 102)
parser.add_argument("--temperature",        default = 0.2)

# box concept methods
parser.add_argument("--method",             default = "uniform")
parser.add_argument("--offset",             default = [-.15, .15])
parser.add_argument("--center",             default =[.1, .2])
parser.add_argument("--entries",            default = 10)
parser.add_argument("--translator",         default = translator)

# hiearchy graph generation
parser.add_argument("--global_feature_dim", default = 66)

config = parser.parse_args(args = [])