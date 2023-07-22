import torch
import torch.nn as nn

import argparse
from config import *

evalparser = argparse.ArgumentParser()
evalparser.add_argument("--device",         default = device)
evalparser.add_argument("--name",           default = "WLK")
evalparser.add_argument("--checkpoint_dir", default = False)
evalparser.add_argument("--")
evalargs = evalparser.parse_args(args = [])

def evaluate_pointcloud(evalmodel, config, args):
    evalmodel = evalmodel.to(args.device)
    pass

if evalargs.checkpoint_dir:
    model = model(config)
else:
    model = model(config)
    model.load_state_dict(torch.load(evalargs.checkpoint_dir))

model = model.to(evalargs.device)

