import warnings
warnings.filterwarnings("ignore")

import torch
import argparse 
import datetime
import time
import sys

from datasets import *

from config import *
from models import *
from visualize.answer_distribution import *
from visualize.concepts.concept_embedding import *

from torch.utils.tensorboard import SummaryWriter
import torchvision
from skimage import color

def get_not_image(B, W = 128, H = 128, C = 3):
    not_image = torch.zeros([B,W,H,C])
    not_image[:,30:50,50:70,:] = 1
    return not_image

def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = True

def log_imgs(imsize,pred_img,clusters,gt_img,writer,iter_):

    batch_size = pred_img.shape[0]
    
    # Write grid of output vs gt 
    grid = torchvision.utils.make_grid(
                          lin2img(torch.cat((pred_img.cpu(),gt_img.cpu()))),
                          normalize=True,nrow=batch_size)

    # Write grid of image clusters through layers
    if clusters is not None:
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
        visualize_image_grid(cluster_imgs[batch_size,...], row = 1, save_name = "val_cluster")
    writer.add_image("Output_vs_GT",grid.detach().numpy(),iter_)
    writer.add_image("Output_vs_GT Var",grid.detach().numpy(),iter_)

    visualize_image_grid(pred_img.reshape(batch_size,imsize,imsize,3)[0,...], row = 1, save_name = "val_recon")


def train_image(train_model, config, args):

    train_model = train_model.to(config.device)
    query = True if args.training_mode in ["joint", "query"] else False
    print("\nstart the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    
    #[setup the training and validation dataset]

    if args.dataset == "PTR":
        train_dataset = PTRData("train", resolution = config.resolution)
        val_dataset =  PTRData("val", resolution = config.resolution)
    if args.dataset == "Toys" :
        if query:
            train_dataset = ToyDataWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = ToyData("train", resolution = config.resolution)
    if args.dataset == "Sprites":
        if query:
            train_dataset = SpriteWithQuestions("train", resolution = config.resolution)
        else:
            train_dataset = SpriteData("train", resolution = config.resolution)
    if args.dataset == "Acherus":
        train_dataset = AcherusDataset("train")


    if args.training_mode == "query":
        freeze_parameters(train_model.scene_perception.backbone)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle); B = args.batch_size

    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    if args.training_mode == "query":alpha = 0
    if args.training_mode == "perception":beta = 0
    

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    concept_visualizer = ConceptEmbeddingVisualizer(0, writer)
    recon_flag = True;

    for epoch in range(args.epoch):
        B = args.batch_size
        epoch_loss = 0
        for sample in dataloader:
            
            # [perception module training]
            gt_ims = torch.tensor(sample["image"].numpy()).float().to(config.device)

            outputs = train_model.scene_perception(gt_ims)

            # get the components
            try:
                recons, clusters, all_losses = outputs["recons"],outputs["clusters"],outputs["losses"]
            except:
                recons = None; all_losses = {}; clusters = None

            try:masks = outputs["abstract_scene"][-1]["masks"].permute([0,3,1,2]).unsqueeze(-1)
            except:
                masks = outputs["abstract_scene"][-1]["masks"]
                recons = outputs["full_recons"]

            perception_loss = 0

            betas = [1.0,1.0]
            pred_img = recons
            #pred_img = outputs["full_recons"]
            #if recons is not None: 
            #    for i,pred_img in enumerate(recons[:]):
            perception_loss += torch.nn.functional.l1_loss(pred_img.flatten(), gt_ims.flatten()) * 1
            pred_img = recons

            if "full_recons" in outputs: perception_loss += torch.nn.functional.l1_loss(outputs["full_recons"].flatten(), gt_ims.flatten())
            

            # [language query module training]
            language_loss = 0

            if query:
                for question in sample["question"]:
                    for b in range(len(question["program"])):
                        program = question["program"][b] # string program
                        answer  = question["answer"][b]  # string answer

                        abstract_scene  = outputs["abstract_scene"]
                        top_level_scene = abstract_scene[-1]

                        working_scene = [top_level_scene]
                        
                        scores   = scores = outputs["abstract_scene"][-1]["scores"]
                        EPS = 1e-5
                        scores   = torch.clamp(scores, min = EPS, max = 1 - EPS).reshape([-1])
                        #scores = scores.unsqueeze(0)


                        features = top_level_scene["features"][b].reshape([scores.shape[0],-1])


                        edge = 1e-5
                        if config.concept_type == "box":
                            features = torch.cat([features,edge * torch.ones(features.shape)],-1)#.unsqueeze(0)

                        kwargs = {"features":features,
                                  "end":scores }

                        q = train_model.executor.parse(program)
                        
                        o = train_model.executor(q, **kwargs)
                        #print("Batch:{}".format(b),q,o["end"],answer)
                        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
                        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))

                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss += 0 #+F.mse_loss(int_num + 1,o["end"])
   
                            if itrs % args.checkpoint_itrs == 0:
                                #print(q,answer)
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).cpu().detach())
                                answer_distribution_num(o["end"].cpu().detach().numpy(),1+int_num.cpu().detach().numpy())
                        if answer in yes_or_no:
                            if answer == "yes":language_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
        
                            if itrs % args.checkpoint_itrs == 0:
                                #print(q,answer)
                                #print(torch.sigmoid(o["end"]).cpu().detach().numpy())
                                visualize_scores(scores.reshape([args.batch_size,-1,1]).cpu().detach())
                                answer_distribution_binary(torch.sigmoid(o["end"]).cpu().detach().numpy())
            # [calculate the working loss]
            working_loss = perception_loss * alpha + language_loss * beta
            epoch_loss += working_loss.detach().cpu().numpy()

            # [backprop and optimize parameters]
            for i,losses in enumerate(all_losses):
                for loss_name,loss in losses.items():
                    writer.add_scalar(str(i)+loss_name, loss, itrs)

            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss, itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs):
                num_concepts = 8

                name = args.name
                expr = args.training_mode
                num_slots = masks.shape[1]
                torch.save(train_model.state_dict(), "checkpoints/{}_{}_{}_{}.pth".format(name,expr,config.domain,config.perception))

                if pred_img is None: pred_img = get_not_image(B)
                pred_img = pred_img.reshape(B,128**2,3)


                log_imgs(config.imsize,pred_img.cpu().detach(), clusters, gt_ims.reshape([args.batch_size,config.imsize ** 2,3]).cpu().detach(),writer,itrs)
                
                visualize_image_grid(gt_ims.flatten(start_dim = 0, end_dim = 1).cpu().detach(), row = args.batch_size, save_name = "ptr_gt_perception")
                visualize_image_grid(gt_ims[0].cpu().detach(), row = 1, save_name = "val_gt_image")

                # * gt_im

                single_comps =  torchvision.utils.make_grid((masks )[0:1].cpu().detach().permute([0,1,4,2,3]).flatten(start_dim = 0, end_dim = 1),normalize=True,nrow=num_slots).permute(1,2,0)
                visualize_image_grid(single_comps.cpu().detach(), row = 1, save_name = "slot_masks")
                #visualize_psg(gt_ims[0:1].cpu().detach(), outputs["abstract_scene"], args.effective_level)

            itrs += 1

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Percept:{} Language:{}, Time: {}".format(epoch + 1, itrs, working_loss,perception_loss,language_loss,datetime.timedelta(seconds=time.time() - start)))
        writer.add_scalar("epoch_loss", epoch_loss, epoch)
    print("\n\nExperiment {} : Training Completed.".format(args.name))


def train_physics(train_model, config , args):
    if args.dataset == "physica":
        train_dataset = PhysicaDataset(config)
    if args.dataset == "industry":
        train_dataset = IndustryDataset(config)

    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)

    dataloader = DataLoader(train_dataset, shuffle = True, batch_size = config.batch_size)
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    for epoch in range(config.epochs):
        epoch_loss = 0
        for sample in dataloader:
            itrs += 1

            # [Collect Physics Inputs]
            sample = sample["physics"] 
            unary_inputs = sample["unary"] # [B,T,N,Fs] 
            binary_relations = sample["binary"] # [B,T,N,N,Fr]

            outputs = train_model(sample)

            trajectory = outputs["trajectory"]
            losses = outputs["losses"]

            # [Weight each component of Working loss]
            working_loss = 0
            for loss_name in losses:
                loss_value = losses[loss_name]
                if loss_name in losses: loss_weight = args.weights[loss_name]
                else: loss_weight = 1.0
                working_loss += loss_value * loss_weight
                writer.add_scalar(loss_name, loss_value, itrs)
            
            # [Optimize Physics Model Parameters]
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()
            
            # [Calcualte the Epoch Loss and ETC]
            epoch_loss += working_loss.detach()
            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Time: {}"\
                .format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))
            

    print("\n\nExperiment {}: Physics Training Completed.".format(args.name))