import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
from matplotlib.patches import Rectangle

def visualize_image_grid(images, row, save_name = "image_grid"):
    plt.figure(save_name, frameon = False);plt.cla()
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    comps_grid = torchvision.utils.make_grid(images,normalize=True,nrow=row).permute(1,2,0)
    
    plt.imshow(comps_grid.cpu().detach().numpy())
    plt.savefig("outputs/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)



def visualize_psg(gt_img, scene_tree, effective_level = 1,):
    scale = gt_img.shape[1]
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[effective_level]["masks"]

        visualize_scores(scene_tree[effective_level]["masks"][0:1].cpu().detach(),"{}".format(level_idx) )

    if effective_level == 1:
        masks = scene_tree[effective_level]["masks"]
        for j in range(masks.shape[1]):
            save_name = "mask_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

            match = scene_tree[effective_level]["match"][0].detach().numpy()
            centers = scene_tree[effective_level - 1]["centroids"][0].detach().numpy()
            moments = scene_tree[effective_level - 1]["moments"][0].detach().numpy()
            plt.imshow(gt_img[0])

            """
            for k in range(match.shape[0]):
                center = centers[k]
                moment = moments[k]
                match_score = float(match[k,j].cpu().detach().numpy())

                lower_x = center[0] * scale 
                lower_y = scale - center[1] * scale

                x_edge = moment[0] * scale /2
                y_edge = moment[1] * scale /2
            """

            plt.scatter(centers[:,0] * scale, scale -centers[:,1] * scale, alpha = match[:,j], color = "purple")

            """
                plt.gca().add_patch(Rectangle((lower_x - x_edge,lower_y - y_edge),2 * x_edge,2 *y_edge,
                    alpha = 1.0,
                    edgecolor='red',
                    facecolor='red',
                    lw=4))
            """
                
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)



def visualize_scene(gt_img, scene_tree, effective_level = 0):
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[effective_level]["masks"]

        visualize_scores(scene_tree[effective_level]["masks"][0:1].cpu().detach(),"{}".format(level_idx) )

def visualize_tree(scores, connections, scale = 1.2, file_name = "temp.png", c = "black"):
    fig = plt.figure("tree-visualize",frameon = False)
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.axis("off")
    x_locs = []; y_locs = []
    for i,score in enumerate(reversed(scores)):
        num_nodes = len(score)
        # calculate scores each node
        #print(score.sigmoid())
        #score = (score.sigmoid() + 0.5).int()
        #scores = score.sigmoid()
        score = torch.clamp(score,0.05,1)

        y_positions = [-scale*(i+1) / 2.0] * num_nodes
        x_positions = np.linspace(-scale**(i+1), scale**(i+1), num_nodes)
        if num_nodes == 1: x_positions = [0.0]
        x_locs.append(x_positions); y_locs.append(y_positions)
        
        plt.scatter(x_positions, y_positions, alpha = score.cpu().detach(), color = c, linewidths=2.0)
    for k,connection in enumerate(reversed(connections)):
        connection = connection
        
        lower_node_num = len(x_locs[k])
        upper_node_num = len(x_locs[k+1])
        for i in range(lower_node_num):
            for j in range(upper_node_num):
                plt.plot( [x_locs[k][i],x_locs[k+1][j]], [y_locs[k][i], y_locs[k+1][j]], color = c ,alpha = float(connection[j][i]))
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
    plt.savefig(file_name)

def visualize_tree_legacy(gt_img,scene_tree,effective_level):
    """
    only visualize single image case!
    """
    assert len(scene_tree) >= effective_level,print("Effective Level Larger than Scene Graph")
    for i in range(effective_level):
        level_idx = effective_level - i
        masks = scene_tree[level_idx]["local_masks"]

        for j in range(masks.shape[1]):
            save_name = "mask_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

  
            plt.imshow(masks[0,j,...].cpu().detach().numpy(), cmap="bone")
            
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)
    
            save_name = "comp_{}_{}".format(level_idx,j+1)
            plt.figure(save_name, frameon = False);plt.cla()
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

            plt.imshow(gt_img[0]/255 * (masks[0,j,...].unsqueeze(-1).cpu().detach().numpy()) )
            plt.savefig("outputs/details/{}.png".format(save_name), bbox_inches='tight', pad_inches=0)

            visualize_scores(scene_tree[effective_level]["masks"][0:1])
        

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

def visualize_scores(scores, name = "set"):
    batch_size = scores.shape[0]
    score_size = scores.shape[1]

    row = batch_size * score_size 
    col = row / 4

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

    plt.savefig("outputs/scores_{}.png".format(name))

def answer_distribution_num(count, target, name = "answer_distribution"):
    batch_size = 1
    score_size = 4

    row = batch_size * score_size 
    col = row / 2

    plt.figure("dists",frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)
    plt.cla()

    x = np.linspace(0,5,100)
    y = np.exp( 0 - (x-target) * (x-target) / 2)
    plt.plot(x,y)
    plt.scatter(target,1)
    plt.scatter(count,np.exp( 0 - (target-count) * (target-count) / 2))
    
    plt.savefig("outputs/{}.png".format(name))
    


def answer_distribution_binary(score, name = "answer_distribution"):
    batch_size = 1
    score_size = 4

    row = batch_size * score_size 
    col = row / 2

    scores = [score, 1 - score]

    plt.figure("dists", frameon = False, figsize = (row,col))
    plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)
    plt.cla()
    
    for i in range(batch_size):
        plt.subplot(1,batch_size,i + 1,frameon=False)
        plt.cla()

        
        keys = ["yes","no"]
        plt.bar(keys,scores)
        plt.tick_params(left = True, right = False , labelleft = True ,
                labelbottom = True, bottom = True)

    plt.savefig("outputs/{}.png".format(name))

# From SRN utils, just formats a flattened image for image writing
def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

# Takes the pred img and clusters produced and writes them to a TF writer


def get_prob(executor,feat,concept):
        pdf = []
        for predicate in executor.concept_vocab:
            pdf.append(torch.sigmoid(executor.entailment(feat,
                executor.get_concept_embedding(predicate) )))
        pdf = torch.cat(pdf, dim = 0)
        idx = executor.concept_vocab.index(concept)
        return pdf[idx]/ pdf.sum(dim = 0)

def build_label(feature, executor):
    default_label = "x"
    default_color = [0,0,0,0.1]
    predicates = executor.concept_vocab
    prob = 0.0
    
    for predicate in predicates:
        pred_prob = get_prob(executor, feature, predicate)
        if pred_prob > prob:
            prob = pred_prob
            default_label = predicate
            #default_label = "{}_{:.2f}".format(predicate,float(pred_prob))

    default_color = [1,0,0.4,float(prob)]
    return default_label, default_color

def visualize_output_trees(scores, features, connections,executor, kwargs):
    plt.cla()
    shapes = [score.shape[0] for score in scores]
    nodes = [];labels = [];colors = [];layouts = []
    # Initialize Scores
    width = 0.9; height = 1.0
    plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)

    for i in range(len(scores)):
        if len(scores[i]) == 1: xs = [0.0];
        else: xs = torch.linspace(-1,1,len(scores[i])) * (width ** i)
        for j in range(len(scores[i])):
            nodes.append(sum(shapes[:i]) + j)
            
            if len(features[i].shape) == 3:
                label,c = build_label(features[i][:,j], executor)
            else:label,c = build_label(features[i][j], executor)
            c[0] = float(torch.linspace(0,1,len(scores))[i])
            c[-1] = min(float(scores[i][j]),c[-1])
            c[-1] = min(float(scores[i][j]),c[-1])
            label = "{}_{:.2f}".format(label,c[-1])
            labels.append(label);colors.append(c)
            # layout the locations
            layouts.append([xs[j],i * height])
            plt.scatter(xs[j],i * height,color = c, linewidths=10)
            plt.text(xs[j], i * height - 0.01, label)

    for n in range(len(connections)):
        connection = connections[n].permute(1,0)

        for i in range(len(connection)):
            for j in range(len(connection[i])):                
                u = i + sum(shapes[:n])
                v = j + sum(shapes[:n+1])
       
                plt.plot(
                    (layouts[u][0],layouts[v][0]),
                    (layouts[u][1],layouts[v][1]), alpha = float(connection[i][j].detach()), color = "black")

    return 