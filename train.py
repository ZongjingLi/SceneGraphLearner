import torch
import argparse 
import datetime
import time
import sys

from datasets.ptr import *

from config import *
from models import *
from visualize.answer_distribution import visualize_image_grid

def train(model, config, args):
    print("\nstart the experiment: {}".format(args.name))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    if args.dataset == "ptr":
        train_dataset = PTRData("train", resolution = config.resolution)
        train_dataset =  PTRData("val", resolution = config.resolution)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle)

    itrs = 0
    start = time.time()

    for epoch in range(args.epoch):
        for sample in dataloader:
            
            ims = sample["image"]

            working_loss = 0

            if not(itrs % args.checkpoint_itrs):
                visualize_image_grid(ims.flatten(start_dim = 0, end_dim = 1).cpu().detach(), row = args.batch_size, save_name = "ptr_gt_perception")
            itrs += 1

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {}, Time: {}".format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))
    
    print("\n\nExperiment {} : Training Completed.".format(args.name))



argparser = argparse.ArgumentParser()
# [general config of the training]
argparser.add_argument("--name",                    default = "KFT")
argparser.add_argument("--epoch",                   default = 1)
argparser.add_argument("--optimizer",               default = "Adam")
argparser.add_argument("--lr",                      default = 2e-4)
argparser.add_argument("--batch_size",              default = 4)
argparser.add_argument("--dataset",                 default = "ptr")

# [additional training details]
argparser.add_argument("--warmup",                  default = False)
argparser.add_argument("--warmup_steps",            default = 1000)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 1)

args = argparser.parse_args(args = [])

if args.checkpoint_dir:
    model = torch.load(args.checkpoint_dir, map_location = config.device)
else:
    print("No checkpoint to load and creating a new model instance")
    model = SceneLearner(config)

train(model, config, args)

