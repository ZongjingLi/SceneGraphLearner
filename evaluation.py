import torch
import torch.nn as nn

import argparse
from datasets import *
from config import *

evalparser = argparse.ArgumentParser()
evalparser.add_argument("--device",         default = device)
evalparser.add_argument("--name",           default = "WLK")
evalparser.add_argument("--checkpoint_dir", default = False)
evalparser.add_argument("--")
evalargs = evalparser.parse_args(args = [])

def evaluate_pointcloud(evalmodel, config, args, phase = "0"):
    evalmodel = evalmodel.to(args.device)
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = True if args.training_mode in ["joint", "query"] else False

    if args.dataset == "Objects3d":
        eval_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))

    evalloader = DataLoader(eval_dataset)
    for sample in evalloader:
        inputs = sample
        outputs = evalmodel(inputs)

    print("evaluation process done!")

if evalargs.checkpoint_dir:
    model = model(config)
else:
    model = model(config)
    model.load_state_dict(torch.load(evalargs.checkpoint_dir))

model = model.to(evalargs.device)

