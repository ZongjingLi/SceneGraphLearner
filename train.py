import torch
import argparse 
import datetime
import time
import sys

from datasets.ptr import *

from config import *
from models import *
from visualize.answer_distribution import visualize_image_grid,lin2img

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color


def log_imgs(imsize,pred_img,clusters,gt_img,writer,iter_):

    batch_size = pred_img.shape[0]
    
    # Write grid of output vs gt 
    grid = torchvision.utils.make_grid(
                          lin2img(torch.cat((pred_img.cpu(),gt_img.cpu()))),
                          normalize=True,nrow=batch_size)

    # Write grid of image clusters through layers
    cluster_imgs = []
    for i,(cluster,_) in enumerate(clusters):
        for cluster_j,_ in reversed(clusters[:i+1]): cluster = cluster[cluster_j]
        pix_2_cluster = to_dense_batch(cluster,clusters[0][1])[0]
        cluster_2_rgb = torch.tensor(color.label2rgb(
                    pix_2_cluster.detach().cpu().numpy().reshape(-1,imsize,imsize) 
                                    ))
        cluster_imgs.append(cluster_2_rgb)
    cluster_imgs = torch.cat(cluster_imgs)
    grid2=torchvision.utils.make_grid(cluster_imgs.permute(0,3,1,2),nrow=batch_size)
    writer.add_image("Clusters",grid2.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT",grid.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT Var",grid.detach().numpy(),iter_)

    visualize_image_grid(cluster_imgs[0,...], row = 1, save_name = "val_cluster")
    visualize_image_grid(pred_img.reshape(batch_size,imsize,imsize,3)[0,...], row = 1, save_name = "val_recon")

def train(model, config, args):
    print("\nstart the experiment: {}".format(args.name))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    #[setup the training and validation dataset]
    if args.dataset == "ptr":
        train_dataset = PTRData("train", resolution = config.resolution)
        val_dataset =  PTRData("val", resolution = config.resolution)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle)

    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    
    query = True if args.training_mode in ["joint", "query"] else False

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    for epoch in range(args.epoch):
        for sample in dataloader:
            
            # [perception module training]
            gt_ims = torch.tensor(sample["image"].numpy()).float().to(config.device)

            outputs = model.scene_perception(gt_ims)
            recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]


            perception_loss = 0

            for i,pred_img in enumerate(recons[:]):
                perception_loss += torch.nn.functional.l1_loss(pred_img.flatten(), gt_ims.flatten())
            

            # [language query module training]
            language_loss = 0
            if query:
                language_loss += 0

            # [calculate the working loss]
            working_loss = perception_loss * alpha + language_loss * beta

            # [backprop and optimize parameters]
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            for i,losses in enumerate(all_losses):
                for loss_name,loss in losses.items():
                    writer.add_scalar(str(i)+loss_name, loss, itrs)
            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss, itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs):
                log_imgs(config.imsize,pred_img.cpu().detach(), clusters, gt_ims.reshape([args.batch_size,config.imsize ** 2,3]).cpu().detach(),writer,itrs)
                
                visualize_image_grid(gt_ims.flatten(start_dim = 0, end_dim = 1).cpu().detach(), row = args.batch_size, save_name = "ptr_gt_perception")
                visualize_image_grid(gt_ims[0].cpu().detach(), row = 1, save_name = "val_gt_image")

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

# [perception and language grounding training]
argparser.add_argument("--training_mode",           default = "perception")
argparser.add_argument("--alpha",                   default = 1.0)
argparser.add_argument("--beta",                    default = 1.0)

# [additional training details]
argparser.add_argument("--warmup",                  default = False)
argparser.add_argument("--warmup_steps",            default = 1000)
argparser.add_argument("--decay",                   default = False)
argparser.add_argument("--decay_steps",             default = 20000)
argparser.add_argument("--decay_rate",              default = 0.99)
argparser.add_argument("--shuffle",                 default = True)

# [checkpoint location and savings]
argparser.add_argument("--checkpoint_dir",          default = False)
argparser.add_argument("--checkpoint_itrs",         default = 10)
argparser.add_argument("--pretrain_perception",     default = False)

args = argparser.parse_args(args = [])

if args.checkpoint_dir:
    model = torch.load(args.checkpoint_dir, map_location = config.device)
else:
    print("No checkpoint to load and creating a new model instance")
    model = SceneLearner(config)

if args.pretrain_perception:
    model.scene_perception = torch.load(args.pretrain_perception, map_location = config.device)

train(model, config, args)

