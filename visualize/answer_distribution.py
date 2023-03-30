import numpy as np
import matplotlib.pyplot as plt

import torchvision

def visualize_outputs(image, outputs):

    full_recon = outputs["full_recons"]
    recons     = outputs["recons"]
    masks      = outputs["masks"]

    num_slots = recons.shape[1]
    
    # [Draw Components]
    plt.figure("Components",frameon=False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    comps_grid = torchvision.utils.make_grid((recons*masks).cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    
    plt.imshow(comps_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/components.png")

    # [Draw Masks]
    plt.figure("Masks",frameon=False);plt.cla()
    masks_grid = torchvision.utils.make_grid(masks.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(masks_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/masks.png")

    # [Draw Recons]
    plt.figure("Recons",frameon=False);plt.cla()
    recon_grid = torchvision.utils.make_grid(recons.cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(recon_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/recons.png")

    # [Draw Full Recon]
    plt.figure("Full Recons",frameon=False);plt.cla()
    grid = torchvision.utils.make_grid(full_recon.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=1)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.imshow(grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/full_recons.png")

    # [Draw GT Image]
    plt.figure("GT Image",frameon=False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    gt_grid = torchvision.utils.make_grid(image.cpu().detach().permute([0,3,1,2]),normalize=True,nrow=1)
    plt.imshow(gt_grid.permute(1,2,0).cpu().detach().numpy())
    plt.savefig("outputs/gt_image.png")

def visualize_distribution(values):
    plt.figure("answer_distribution", frameon = False)
    plt.cla()
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = False)
    keys = list(range(len(values)))
    plt.bar(keys,values)

def visualize_scores(scores):
    batch_size = scores.shape[0]
    score_size = scores.shape[1]

    row = batch_size * score_size / 2
    col = row / 8

    plt.figure("scores", frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = False, bottom = False)
    plt.cla()
    
    for i in range(batch_size):
        plt.subplot(1,batch_size,i + 1,frameon=False)
        plt.cla()

        
        keys = list(range(score_size))
        plt.bar(keys,scores[i])
        plt.tick_params(left = False, right = False , labelleft = True ,
                labelbottom = False, bottom = False)

    plt.savefig("outputs/scores.png")

