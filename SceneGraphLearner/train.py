import warnings

from env.mkgrid.northrend_env import *
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
from visualize import *
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

def build_scene_tree(perception_outputs):
    all_kwargs = []
    scene_tree = perception_outputs["scene_tree"]
    scores = scene_tree["object_scores"]
    features = scene_tree["object_features"]
    connections = scene_tree["connections"]

    B = features[0].shape[0]
    for b in range(B):
        kw_scores, kw_features, kw_connections = [score[b] for score in scores], [feature[b] for feature in features], \
        [connection[b] for connection in connections]
        kwargs = {"features":kw_features, "end":kw_scores, "connections":kw_connections}
        all_kwargs.append(kwargs)
    return all_kwargs

def collect_qa_batch(batch_qa):
    batch_wise_qa = []
    
    for b in range(len(batch_qa[0]["program"])):
        questions_in_batch = {"question":[],"program":[],"answer":[]}
        for qpair in batch_qa:

            try:questions_in_batch["question"].append(qpair["question"][b])
            except: questions_in_batch["question"].append("not generated yet.")
            questions_in_batch["program"].append(qpair["program"][b])
            questions_in_batch["answer"].append(qpair["answer"][b])
        batch_wise_qa.append(questions_in_batch)
    return batch_wise_qa


def train_scenelearner(train_model, config, args):
    root = config.root
    IGNORE_KEY = True
    train_model = train_model.to(config.device)
    query = True if args.training_mode in ["joint","query"] else False
    print("\nstart the experiment: {} query:[{}]".format(args.name,query))
    print("\nTrain the Scene Learner Model-[{}] on [{}]".format(args.perception,args.dataset))

    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    

    # [Create the Dataloader]
    if args.dataset == "Hearth":
        train_dataset = HearthDataset("train", resolution = config.resolution)
    if args.dataset == "Battlecode":
        train_dataset = BattlecodeImageData("train", resolution = config.resolution)
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
        freeze_parameters(train_model.scene_perception)

    dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = args.shuffle); B = args.batch_size

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./tf-logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)
    
    for epoch in range(args.epoch):
        B = args.batch_size
        epoch_loss = 0.
        for sample in dataloader:
            input_images = sample["image"]
            actual_batch_size = input_images.shape[0]
            # [Perform Image Level Perception]
            perception_outputs = train_model.scene_perception(input_images) # 

            # [Calculate the Perception Loss]
            perception_loss = 0.0
            perception_losses = perception_outputs["losses"]
            for loss_name in perception_losses:
                loss_item = perception_losses[loss_name]
                if IGNORE_KEY:
                    if loss_name in args.loss_weights:loss_weight = args.loss_weights[loss_name]
                    else:loss_weight = 1.0
                else:  loss_weight = args.loss_weights[loss_name]
                writer.add_scalar(loss_name,loss_item, itrs)
                perception_loss += loss_item * loss_weight

            # [Calculate the Language Loss]
            language_loss = 0.0
            if query: # do something about the query
                # [Visualize Predicate Segmentation]
                batch_qa_data = collect_qa_batch(sample["question"])
                batch_kwargs = build_scene_tree(perception_outputs)
                
                qa_summary = ""

                for b in range(input_images.shape[0]):
                    qa_summary += "\n\nBatch:{}\n".format(b)
                    questions = batch_qa_data[b]["question"]
                    programs = batch_qa_data[b]["program"]
                    answers = batch_qa_data[b]["answer"]
                    kwargs = batch_kwargs[b]
                    batch_total_qa = 0
                    batch_correct_qa = 0
                    for i,q in enumerate(questions):
                        qa_summary += "\n"
                        question = questions[i]
                        q = programs[i]
                        q = train_model.executor.parse(q)
                        o = train_model.executor(q, **kwargs)
                        answer = answers[i]
                        qa_summary += "\n" + question + "\n"
                        qa_summary += programs[i] + "\n"

                        if answer in ["True","False"]:answer = {"True":"yes","False":"no"}[answer]
                        if answer in ["1","2","3","4","5","6","7","8","9"]:answer = num2word(int(answer))

                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss +=  F.mse_loss(int_num,o["end"])
                            predict_answer = answer#num2word(int(o["end"]))

                            if itrs % args.checkpoint_itrs == 0:
                                answer_distribution_num(o["end"].cpu().detach().numpy(),int_num.cpu().detach().numpy())

                        if answer in yes_or_no:
                            if answer == "yes":language_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
                            if torch.sigmoid(o["end"]) > 0.5:
                                predict_answer = "yes"
                            else:predict_answer = "no"
    
                            if itrs % args.checkpoint_itrs == 0:
                                save_name = root+"b{}_q{}_answer_distribution".format(b,q)
                                answer_distribution_binary(o["end"].sigmoid().cpu().detach(),save_name)
                        print(answer)
                        if predict_answer == answer:
                            batch_correct_qa += 1
                        batch_total_qa += 1
                        qa_summary += "\ngt_answer: {}\n\npd_answer:{}\n".format(answer,predict_answer)
                    qa_summary += "\n Acc:{} = {}/{}\n\n".format(float(batch_correct_qa)/float(batch_total_qa),\
                                                        batch_correct_qa,batch_total_qa)
                if itrs % args.checkpoint_itrs == 0:
                    writer.add_text("Language Summary",qa_summary, itrs)
            
            # [Overall Joint Loss of Perception and Language]
            working_loss = perception_loss+ language_loss # calculate the overall working loss of the system
            epoch_loss += working_loss.cpu().detach() # just for statisticss
            
            # [BackProp]
            optimizer.zero_grad()
            working_loss.backward()
            optimizer.step()

            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss, itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs): # [Visualzie Outputs] always visualize all the batch
                scene_tree = perception_outputs["scene_tree"]
                torch.save(train_model,root + "/checkpoints/temp.ckpt")

                scene_tree_ims = []
                for b in range(min(2, actual_batch_size)):
                    vis_scores = [score[b].detach() for score in scene_tree["object_scores"][1:]]
                    vis_connections = [connect[b] for connect in scene_tree["connections"][1:]]
                    visualize_tree(vis_scores, vis_connections, scale = 1.618, file_name = "outputs/scene_tree{}.png".format(b))

                    for i,recon in enumerate(perception_outputs["reconstructions"]):

                        #print(perception_outputs["masks"][i+1][b,...].shape,recon[b,...].shape)
                        n,w,h,_ = recon[b,...].shape
                        #curr_recon = (perception_outputs["masks"][i][b,...].reshape(w,h,n,1).permute(2,0,1,3)*
                        curr_recon = (recon[b,...]).cpu().detach()
                        plt.figure("recon");plt.cla()
                        plt.imshow(recon[b,...].sum(0).detach())
                        plt.savefig("temp.png")
                        save_name = "batch{}_recon_layer{}.png".format(b, i + 1)
                        visualize_image_grid(curr_recon.permute(0,3,1,2), curr_recon.shape[0], save_name)
                        vis_masks = perception_outputs["masks"][i][b,...].reshape(w,h,n).permute(2,0,1)
                        #for i in range(vis_masks.shape[0]):print(vis_masks[i,:,:].max(), vis_masks[i].min())
                        vis_masks = vis_masks.unsqueeze(1)
                        visualize_image_grid(vis_masks, vis_masks.shape[0], "masks")
                        writer.add_images("batch{}_layer{}".format(b, i + 1),curr_recon.permute(0,3,1,2),itrs)
                       
                    scene_tree_ims.append(torch.tensor(plt.imread("outputs/scene_tree{}.png".format(b))).unsqueeze(0))

                visualize_image_grid(input_images.permute(0,3,1,2),B,"input_image.png")
                writer.add_images("input_image",input_images.permute(0,3,1,2),itrs)

                scene_tree_ims = torch.cat(scene_tree_ims, dim = 0)
             
                writer.add_images("scene_tree", scene_tree_ims[:,:,:,:3].permute(0,3,1,2), itrs)
        

            itrs += 1

            sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Percept:{} Language:{}, Time: {}".format(epoch + 1, itrs, working_loss,perception_loss,language_loss,datetime.timedelta(seconds=time.time() - start)))
        writer.add_scalar("epoch_loss", epoch_loss, epoch)

    print("\n\nExperiment {} : Training Completed.".format(args.name))


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

def train_pointcloud(train_model, config, args, phase = "1"):
    B = int(args.batch_size)
    train_model = train_model.to(config.device)
    train_model.config = config
    train_model.executor.config = config
    assert phase in ["0", "1", "2", "3", "4", "5",0,1,2,3,4,5],print("not a valid phase")
    query = False if args.phase in ["0",0] else True
    clip_grads = query
    print("\nstart the experiment: {} query:[{}]".format(args.name,query))
    print("experiment config: \nepoch: {} \nbatch: {} samples \nlr: {}\n".format(args.epoch,args.batch_size,args.lr))
    if args.phase in ["1",1]: args.loss_weights["equillibrium"] = 0.01
    #[setup the training and validation dataset]

    if args.dataset == "Objects3d":
        train_dataset= Objects3dDataset(config, sidelength = 128, stage = int(phase))
    if args.dataset == "StructureNet":
        if args.phase in ["0",]:
            train_dataset = StructureDataset(config, category = "vase")
        if args.phase in ["1","2","3","4"]:
            train_dataset = StructureGroundingDataset(config, category = args.category, split = "train", phase = "1")
    
    #train_dataset = StructureGroundingDataset(config, category="vase", split = "train")
    dataloader = DataLoader(train_dataset, batch_size = int(args.batch_size), shuffle = args.shuffle)


    # [joint training of perception and language]
    alpha = args.alpha
    beta  = args.beta
    if args.training_mode == "query":alpha = 1
    if args.training_mode == "perception":beta = 1
    

    # [setup the optimizer and lr schedulr]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(train_model.parameters(), lr = args.lr)
    if args.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(train_model.parameters(), lr = args.lr)

    if args.freeze_perception:
         print("freezed the perception module: True")
         freeze_parameters(train_model.part_perception)
    if phase not in ["0"]:
       freeze_hierarchy(train_model,int(phase))
         
    # [start the training process recording]
    itrs = 0
    start = time.time()
    logging_root = "./logs"
    ckpt_dir     = os.path.join(logging_root, 'checkpoints')
    events_dir   = os.path.join(logging_root, 'events')
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    if not os.path.exists(events_dir): os.makedirs(events_dir)
    writer = SummaryWriter(events_dir)
    max_gradient = 1000.
    for epoch in range(args.epoch):
        epoch_loss = 0
        for sample in dataloader:
            sample, gt = sample
            # [perception module training]

            outputs = train_model.part_perception(sample)
            all_losses = {}

            # [Perception Loss]
            perception_loss = 0
            for loss_name in outputs["loss"]:
                weight = args.loss_weights[loss_name]
                perception_loss += outputs["loss"][loss_name] * weight
            components = outputs["components"]

            # [Language Loss]
            language_loss = 0
            if query:
                features,masks,positions = outputs["features"],outputs["masks"],outputs["positions"] 

                qa_programs = sample["programs"]
                answers = sample["answers"]

                features = train_model.feature2concept(features)
                if config.concept_type == "box" and False:

                    features = torch.cat([
                        features, 
                        EPS * torch.ones(B,features.shape[1],config.concept_dim)\
                        ],dim = -1)
                #print(features.shape)
                scene = train_model.build_scene(features)

            
                for b in range(features.shape[0]):
                    scores,features,connections = load_scene(scene, b)

                    kwargs = {"features":features,
                    "end":scores,
                    "connections":connections}

                    for i,q in enumerate(qa_programs):
                        answer = answers[i][b]

                        q = train_model.executor.parse(q[0])

                        
                        o = train_model.executor(q, **kwargs)
                        

                        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
                        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))
                        
                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss += F.mse_loss(int_num ,o["end"])
                            
                        if answer in yes_or_no:
                            if answer == "yes":
                                language_loss -= torch.log(torch.sigmoid(o["end"]))
                            else:
                                language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
                        #language_loss += (1 + torch.sigmoid( (o["end"] + g) )/r)

            # [calculate the working loss]

            working_loss = perception_loss + language_loss
            try:epoch_loss += working_loss.detach().numpy()
            except:epoch_loss += working_loss
            # [backprop and optimize parameters]
            for i,losses in enumerate(all_losses):
                for loss_name,loss in losses.items():
                    writer.add_scalar(str(i)+loss_name, loss, itrs)

            optimizer.zero_grad()
            working_loss.backward()
            if clip_grads and 0:
                torch.nn.utils.clip_grad_norm(train_model.executor.parameters() , -max_gradient, max_gradient)
            optimizer.step()

            writer.add_scalar("working_loss", working_loss, itrs)
            writer.add_scalar("perception_loss", perception_loss , itrs)
            writer.add_scalar("language_loss", language_loss, itrs)

            if not(itrs % args.checkpoint_itrs):
                if query:
                    print("")
                    for i,q in enumerate(qa_programs):
                        answer = answers[i][b]
                        q = train_model.executor.parse(q[0])
                        o = train_model.executor(q, **kwargs)
                        if answer in ["True","False"]:answer = {"True":"yes,","False":"no"}[answer]
                        if answer in ["1","2","3","4","5"]:answer = num2word(int(answer))
                        
                        if answer in numbers:
                            int_num = torch.tensor(numbers.index(answer)).float().to(args.device)
                            language_loss += +F.mse_loss(int_num ,o["end"])
                            print(q,answer,"mse:",float(F.mse_loss(int_num,o["end"]).detach().numpy()))
                        if answer in yes_or_no:
                            if answer == "yes":
                                language_loss -= torch.log(torch.sigmoid(o["end"]))
                                print(q,answer,"p:",float(torch.sigmoid(o["end"])))
                            else:
                                language_loss -= torch.log(1 - torch.sigmoid(o["end"]))
                                print(q,answer,"p:",float(1 - torch.sigmoid(o["end"])))

                    scene_tree_path = sample["scene_tree"][-1]
                    scene_tree = nx.read_gpickle(scene_tree_path)
                    plt.figure("Comparison",figsize=(14,6))
                    plt.subplot(121)
                    plt.cla()
                    pos = nx.layout.spring_layout(scene_tree)
                    nx.draw_networkx(scene_tree,pos)
                    plt.subplot(122)
                    visualize_output_trees(scores, features, connections, model.executor, kwargs)
                    plt.pause(0.001)

                name = args.name
                expr = args.training_mode
                field_class = "3dpc"
                if 1:
                    torch.save(train_model.state_dict(),\
                        "checkpoints/scenelearner/{}/{}_{}_{}_{}_phase{}.pth".format(field_class,name,expr,config.domain,config.perception,phase))
                    torch.save(train_model,\
                        "checkpoints/scenelearner/{}/{}_{}_{}_{}_phase{}.ckpt".format(field_class,name,expr,config.domain,config.perception,phase))
                else:
                    torch.save(train_model.part_perception.state_dict(),"checkpoints/{}_part_percept_{}_{}_{}_phase{}.pth".format(name,expr,config.domain,config.perception,phase))
                if args.dataset == "Objects3d":
                    point_cloud = sample["point_cloud"]
                    rgb = sample["rgb"]
                    coords = sample["coords"]
                    occ = sample["occ"]
                    coord_color = sample["coord_color"]
                    np.save("outputs/point_cloud.npy",np.array(point_cloud[0,:,:].cpu()))
                    np.save("outputs/rgb.npy",np.array(rgb[0,:,:].cpu()))
                    np.save("outputs/coords.npy",np.array(coords[0,:,:].cpu()))
                    np.save("outputs/occ.npy",np.array(occ[0,:,:].cpu()))
                    np.save("outputs/coord_color.npy",np.array(coord_color[0,:,:].cpu()))

                    recon_occ = outputs["occ"]
                    recon_coord_color = outputs["color"]
                    np.save("outputs/recon_occ.npy",np.array(recon_occ[0,:].cpu().unsqueeze(-1).detach()))
                    np.save("outputs/recon_coord_color.npy",np.array(recon_coord_color[0,:,:].cpu().detach()))
                if args.dataset == "StructureNet":
                    recon_pc = outputs["recon_pc"][0]
                    point_cloud = sample["point_cloud"][0]
                    masks = outputs["masks"][0]
                    np.save("outputs/recon_point_cloud.npy",np.array(recon_pc.cpu().detach()))
                    np.save("outputs/point_cloud.npy",np.array(point_cloud.cpu().detach()))
                    np.save("outputs/masks.npy",np.array(masks.cpu().detach()))

                    if components is not None:
                        np.save("outputs/splits.npy",np.array(components.cpu().detach()))

                    # Save Components
                    
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


def train_rl(model, config, args):
    # [Simulation Environment Setup]
    print("start the reinforcement interaction training.")
    if args.env_name == "Northrend":
        env = Northrend(config, load_map = None)
    # [Optimizer Setup]
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    start = time.time()
    itrs = 0
    for epoch in range(args.epoch):
        # [sample trajectories if necessary]
        trajectory_samples = sample_trajectory(model, env, \
                goal = None, num_samples = args.traj_sample_num, visualize_map= False)
        # [optimizer model parameters]
        planning_loss = 0.0
        for i,loss in enumerate(trajectory_samples[0]):
            reward = trajectory_samples[1][i]
            planning_loss += loss * reward

        # [Working Loss Calculated]
        working_loss = planning_loss
        
        optimizer.zero_grad()
        planning_loss.backward()
        optimizer.step()

        itrs += 1
        sys.stdout.write ("\rEpoch: {}, Itrs: {} Loss: {} Time: {}"\
                .format(epoch + 1, itrs, working_loss,datetime.timedelta(seconds=time.time() - start)))
            
    return model

def  sample_trajectory(model, env, goal = None,num_samples=1, max_steps = 1000, visualize_map = True):
    losses = []
    rewards = []
    for epoch in range(num_samples):
        done = False
        steps = 0
        env.reset()

        # [Build a Plan]
        if goal is not None:model.plan(goal)
        plt.figure("epoch:{}".format(epoch))
        epoch_loss = 0
        epoch_reward = 0
        while not done and steps < max_steps:
            # [Get Current State]
            local_obs, global_obs = env.render()

            # [Action Bases on Current Plan and State]
            action,loss = model.get_action(local_obs)
            update = env.step(action)

            # [Calculate Reward and Epoch Loss]
            reward = update["reward"]
            done = update["done"]
            epoch_reward += reward
            epoch_loss += loss

            # [Visualize Results]
            if visualize_map:
                plt.subplot(121);plt.cla();plt.axis("off");plt.imshow(local_obs)
                plt.subplot(122);plt.cla();plt.axis("off");plt.imshow(global_obs)
                plt.pause(0.01)
            steps += 1
        losses.append(epoch_loss / steps)
        rewards.append(epoch_reward / steps)
    return losses, rewards