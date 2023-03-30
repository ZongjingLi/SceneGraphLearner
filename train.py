import torch
import argparse 
import datetime
import time
import sys

from datasets.ptr import *

from config import *
from visualize.answer_distribution import visualize_image_grid

def train(model, config, args):
    print("\nstart the experiment: {}".format(args.name))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    if args.dataset == "ptr":
        train_dataset = PTRData("train", resolution = config.resolution)

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
            



argparser = argparse.ArgumentParser()
argparser.add_argument("--name",                default = "KFT")
argparser.add_argument("--epoch",               default = 3000)
argparser.add_argument("--lr",                  default = 2e-4)
argparser.add_argument("--batch_size",          default = 4)
argparser.add_argument("--dataset",             default = "ptr")

argparser.add_argument("--checkpoint_itrs",     default = 100)
argparser.add_argument("--shuffle",             default = True)

args = argparser.parse_args(args = [])


train(0, config, args)

