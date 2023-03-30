import torch
import argparse 

from datasets.ptr import *

from config import *

def train(model, config, args):
    print("\nstart the experiment: {}".format(args.name))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    if args.dataset == "ptr":
        train_dataset = PTRData("train", resolution = config.resolution)

argparser = argparse.ArgumentParser()
argparser.add_argument("--name",            default = "KFT")
argparser.add_argument("--epoch",           default = 3000)
argparser.add_argument("--lr",              default = 2e-4)
argparser.add_argument("--batch_size",      default = 4)
argparser.add_argument("--dataset",         default = "ptr")

args = argparser.parse_args(args = [])


train(0, config, args)

