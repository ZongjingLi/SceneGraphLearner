import torch
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("--name",               default = "SceneGraphLearner")

# setup the perception module
parser.add_argument("--resolution",         default = (256,256))
parser.add_argument("--hidden_dim",         default = 100)

# setup the concept learner 
parser.add_argument("--concept_dim",        default = 100)

config = parser.parse_args(args = [])