from ntpath import join
from types import SimpleNamespace

import torch
import torch.nn as nn

from torch_scatter         import scatter_mean

from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max
from models.percept.slot_attention import SlotAttention

from models.percept.stnet.propagation import GraphPropagation

from .convnet               import *
from .affinities            import *
from .primary               import * 
from utils                  import *

import math
import matplotlib.pyplot as plt

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        edges[0].append(loc)
                        edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

def QTR(grid,center,a,ah,aw,ahh,aww,ahw):
    ch,cw = center
    mx,my = grid[:,:,0],grid[:,:,1]
    var = a + \
        ah*(mx - ch) + aw*(my - cw) +\
        ahh*(mx - ch)**2 + aww*(my - cw)**2 + \
            ahw * (mx - ch) * (my - cw)
    return var

def RenderQTR(G,features):

    # Render 20 dim Features using 2 centroids and attribute
    centers   = features[:,:2] # Size:[N,2]
    paras     = features[:,2:] # Size:[N,18]
    mx,my     = G[:,0],G[:,1]
    output_channels = []
    for c in range(3):
        ch,cw = centers[:,0],centers[:,1]
        bp = paras[:,c * 6: (c + 1) * 6]

        a,ah,aw,ahh,aww,ahw = bp[:,0],bp[:,1],bp[:,2],bp[:,3],bp[:,4],bp[:,5]
        qtr_var = a + \
        ah*(mx - ch) + aw*(my - cw) +\
        ahh*(mx - ch)**2 + aww*(my - cw)**2 + \
            ahw * (mx - ch) * (my - cw)
        output_channels.append(qtr_var.unsqueeze(0))

    return torch.cat(output_channels,0).permute([1,0])

def QSR(grid,px,py,pa,pr):
    
    mx,my = grid[:,:,0],grid[:,:,1]
    return torch.sigmoid(\
        pa * (my * (torch.cos(pr)) - mx * (torch.sin(pr)) - px) ** 2 - \
             (mx * (torch.cos(pr)) + my * (torch.sin(pr)) - py)
    )



def render_level(level,name = "Namo",scale = 64):

    plt.scatter(level.centroids[:,0] * scale,(1 - level.centroids[:,1]) * scale,c = "cyan")
    row,col = level.edges

    rc,cc = level.centroids[row] * scale,level.centroids[col] * scale

    for i in range(len(rc)):
        point1 = rc[i];point2 = cc[i]
        x_values = [point1[0], point2[0]]
        y_values = [scale - point1[1], scale - point2[1]]
        plt.plot(x_values,y_values,color = "red",alpha = 0.3)


def optical_flow_motion_mask(video):
    masks = 0
    return masks

class PSGNet(torch.nn.Module):
    def __init__(self,imsize, perception_size, struct = [1, 1]):

        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.imsize = imsize

        node_feat_size   = 64
        num_graph_layers = 2

        
        self.spatial_edges,self.spatial_coords = grid(imsize,imsize,device=device)
        
        # [Global Coords]
        self.global_coords = self.spatial_coords.float()

        self.spatial_edges = build_perception(imsize,perception_size,device = device)
        self.spatial_coords = self.spatial_coords.to(device).float() / imsize
        # Conv. feature extractor to map pixels to feature vectors
        self.rdn = RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))



        # Affinity modules: for now just one of P1 and P2 
        self.affinity_aggregations = torch.nn.ModuleList([])
        for s in struct:
            if s == 1: self.affinity_aggregations.append(P1AffinityAggregation())
            if s == 2: self.affinity_aggregations.append(P2AffinityAggregation())

        # Node transforms: function applied on aggregated node vectors

        self.node_transforms = torch.nn.ModuleList([
            FCBlock(hidden_ch=100,
                    num_hidden_layers=3,
                    in_features =node_feat_size + 4,
                    out_features=node_feat_size,
                    outermost_linear=True) for _ in range(len(self.affinity_aggregations))
        ])

        # Graph convolutional layers to apply after each graph coarsening
        gcv = GraphConv(node_feat_size, node_feat_size)  
        self.graph_convs = torch.nn.ModuleList([
            GraphConv(node_feat_size , node_feat_size ,aggr = "mean")   for _ in range(len(self.affinity_aggregations))
        ])

        # Maps cluster vector to constant pixel color
        self.node_to_rgb  = FCBlock(hidden_ch=100,
                                    num_hidden_layers=3,
                                    in_features =20,
                                    out_features=3,
                                    outermost_linear=True)

        self.node_to_qtr_p1  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.node_to_qtr_p2  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.gauge = nn.Linear(node_feat_size,node_feat_size)


    def dforward(self,imgs,effective_mask = None):
        return 0

    def forward(self,img,effective_mask = None):
        batch_size = img.shape[0]

        mask_shape = [batch_size,self.imsize,self.imsize,1]
        
        # [ Effective Mask Not Considered ]
        if effective_mask is None: effective_mask = torch.ones(mask_shape)


        # [Create the Local Coords]

        # Collect image features with rdn

        im_feats = self.rdn(img.permute(0,3,1,2))
        #im_feats = img.permute(0,3,1,2) 

        #coords_added_im_feats = torch.cat([
        #          self.spatial_coords.unsqueeze(0).repeat(im_feats.size(0),1,1),
        #          im_feats.flatten(2,3).permute(0,2,1)
        #                                  ],dim=2)
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1)

        ### Run image feature graph through affinity modules

        graph_in = Batch.from_data_list([Data(x,self.spatial_edges)
                                                for x in coords_added_im_feats])

        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch

        clusters, all_losses = [], [] # clusters just used for visualizations
        intermediates = [] # intermediate values

        ## Perform Affinity Calculation and Graph Clustering
        level_centroids = []
        level_moments = []
        level_batch = []
        
        for pool, conv, transf in zip(self.affinity_aggregations,
                                      self.graph_convs, self.node_transforms):
            batch_uncoarsened = batch

            x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)
            level_batch.append(batch)

            clusters.append( (cluster, batch_uncoarsened) )
            for i,(cluster_r,_) in enumerate(clusters):
                for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
                device = self.device

                centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device),cluster_r.to(device),dim = 0)
                moments   = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device) ** 2,cluster_r.to(device),dim = 0 )
            
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
    

            # augument each node with explicit calculation of moment and centroids
            x = torch.cat([x,centroids,moments],dim = -1)
            x = transf(x)
            x = conv(x, edge_index)

            all_losses.append(losses)
            intermediates.append(x)

            level_centroids.append(centroids)
            level_moments.append(moments)
            

        joint_spatial_features = []

        for i,(cluster_r,_) in enumerate(clusters):
            for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
            centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1),cluster_r,dim = 0)

            joint_features = torch.cat([self.node_to_qtr_p2(intermediates[i]),centroids],-1)
            joint_spatial_features.append(joint_features)

        recons = [] # perform reconstruction over each layer composed

        for i,jsf in enumerate(joint_spatial_features):
            for cluster,_ in reversed(clusters[:i+1]):jsf = jsf[cluster]

            
            paint_by_numbers = to_dense_batch( 1.0 *  (\
                RenderQTR(self.spatial_coords.repeat(batch_size,1),jsf))\
                    ,graph_in.batch)[0]
            recons.append(paint_by_numbers)
            
        return {"recons":recons,"clusters":clusters,
        "losses":all_losses,
        "features":intermediates, 
        "centroids":level_centroids, 
        "moments":level_moments,
        "batch":level_batch}

def to_dense_features(outputs, size = 128):
    level_batch = outputs["batch"]
    features = outputs["features"]
    centroids = outputs["centroids"]
    moments = outputs["moments"]
    clusters = outputs["clusters"]
    level_features = []
    for i in range(len(features)):
        sparse_feature = features[i]
        sparse_centroid = centroids[i]
        sparse_moment = moments[i]
    
        cast_batch = level_batch[i]

        feature,  batch = to_dense_batch(features[i],cast_batch)
        centroid, batch = to_dense_batch(centroids[i],cast_batch)
        moment,   _batch = to_dense_batch(moments[i],cast_batch)

        """
        cluster_r = clusters[i][0]; 
        batch_r = clusters[i][1]
        for cluster_j,batch_j in reversed(clusters[:i]):
            cluster_r = cluster_r[cluster_j]
            batch_r = batch_r[batch_j]

        cluster_size = int(cluster_r.max()) + 1
        batch_size = int(cast_batch.max()) + 1

        local_masks = torch.zeros([batch_size*size * size, cluster_size])

        local_masks[cluster_r] = 1.0
        local_masks,batch = to_dense_batch(local_masks,batch_r)
        local_masks = local_masks[batch]

        local_masks = local_masks.reshape([batch_size, size, size, cluster_size]).permute([0,3,1,2])
        """

        level_features.append({"features":feature, "centroids":centroid, "moments":moment,"masks":_batch.int(),"local_masks":0})
    return level_features



class AbstractNet(nn.Module):
    def __init__(self,dim = 72, width = 10, iters = 10):
        super().__init__()
        
        self.num_heads = width
        feature_dim = dim
        self.feature_dim = dim
        

        self.propagator = GraphPropagation(num_iters = iters)
        self.feature_heads = nn.Parameter(torch.randn([width, dim]))
        self.spatial_heads = nn.Parameter(torch.randn([width, 2]))

        self.transfer = FCBlock(100,1,dim,dim)


    def forward(self, input_graph):
        # [Feature Propagation]
        
        raw_features =  input_graph["features"]
        B, N, C = raw_features.shape
        temperature = C / C
        raw_spatials =  input_graph["centroids"] * temperature
        masks    =  input_graph["masks"]
        
        B, N, C = raw_features.shape

        # [Build Adjacency Matric]


        adjs = torch.tanh(0.1 * torch.linalg.norm(
         raw_spatials.unsqueeze(1).repeat(1,N,1,1) - 
         raw_spatials.unsqueeze(2).repeat(1,1,N,1), dim = -1) / C) 
        #adjs = torch.zeros([B,N,N,1])

        #TODO: implement a non trivial solution!
        #print("B:{} N:{} C:{}".format(B,N,C))
        """
        plateau_maps = self.propagator(features, adj)
        features = plateau_maps[-1]
        """

        # features after the graph propagation
        if 1:
            self.propagator.num_iters = 15
            joint_feature = torch.cat([raw_features,raw_spatials],-1)

            joint_feature = self.propagator(joint_feature,adjs)[-1]

            features, spatials = torch.split(joint_feature, [C, 2], dim = -1)
        else:
            features, spatials = raw_features, raw_spatials

        

        # [Decode the Matching Head for the input graph]
        # TODO: actually implement a version that is context dependent

        feature_proposals = self.feature_heads.unsqueeze(0).repeat(B, 1, 1)
        spatial_proposals = torch.sigmoid(self.spatial_heads).unsqueeze(0).repeat(B, 1, 1) * temperature

        # [Component Matching]

        # component_features : [B,N,C]
        component_features = torch.cat([features, spatials], -1)

        # feature_proposals  : [B,M,C]
        proposal_features = torch.cat([feature_proposals,spatial_proposals], -1)

 
        match = torch.softmax(masks.unsqueeze(-1) * torch.einsum("bnc,bmc -> bnm",component_features, proposal_features)/math.sqrt(0.1), dim = -1)
    

        output_features = self.transfer(torch.einsum("bnc,bnm->bmc",features, match))
        #print(torch.sum(match,-2))

        existence = torch.max(match, dim = 1).values

        #print("e:",existence)


        out_centroids = torch.einsum("bnk,bnm->bmk",spatials,match)

        # calculate the local mask for visualization

        #input_local_masks = input_graph["local_masks"]
        #print(input_local_masks.shape, match.shape)
        #input_local_masks = torch.ones([B,N,128,128])
        #print(input_local_masks.shape)
        output_local_masks = 0#torch.einsum("bnwh,bnm->bmwh",input_local_masks,match)
        
        output_graph = {"features":output_features, "centroids":out_centroids, "masks":existence, "match":match, "local_masks":output_local_masks, "moments":input_graph["moments"]}
        return output_graph



class FeatureDecoder(nn.Module):
    def __init__(self, inchannel,input_channel,object_dim = 100):
        super(FeatureDecoder, self).__init__()
        self.im_size = 128
        self.conv1 = nn.Conv2d(inchannel + 2, 64, 3, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        # self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(128, 128, 3, bias=False)
        # self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(128, 64, 3, bias=False)
        # self.bn4 = nn.BatchNorm2d(32)
        self.celu = nn.CELU()
        self.celu = nn.Sigmoid()
        self.inchannel = inchannel
        self.conv5_img = nn.Conv2d(64, input_channel, 1)
        self.conv5_mask = nn.Conv2d(64, 1, 1)

        x = torch.linspace(-1, 1, self.im_size + 8)
        y = torch.linspace(-1, 1, self.im_size + 8)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))
        self.bias = 0

        self.object_score_marker  =  nn.Linear(128 * 128 * 64,1)
        #self.object_score_marker   = FCBlock(256,2,64 * 64 * 16,1)
        #self.object_feature_marker = FCBlock(256,3,64 * 64 * 16,object_dim)
        self.object_feature_marker = nn.Linear(128 * 128 * 64,object_dim)
        self.conv_features         = nn.Conv2d(32,16,3,2,1)


    def forward(self, z):
        # z (bs, 32)
        bs,_ = z.shape
       
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size + 8, self.im_size + 8)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(bs, -1, -1, -1),
                       self.y_grid.expand(bs, -1, -1, -1), z), dim=1)
        # x (bs, 32, image_h, image_w)
        x = self.conv1(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -100, max = 100)
        # x = self.bn1(x)
        x = self.conv2(x);x = self.celu(x) * 1.0#self.celu(x)
        x = torch.clamp(x, min = -100, max = 100)
        # x = self.bn2(x)
        x = self.conv3(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -100, max = 100)
        # x = self.bn3(x)
        x = self.conv4(x);x = self.celu(x) * 1.0
        x = torch.clamp(x, min = -100, max = 100)
        #x = self.bn4(x)

        img = self.conv5_img(x)
        img = .5 + 0.5 * torch.tanh(img + self.bias)
        logitmask = self.conv5_mask(x)
        
        #x = self.conv_features(x)
        #python3 train.py --name="KFT" --training_mode="joint" --pretrain_joint="checkpoints/Boomsday_toy_slot_attention.ckpt" --"save_path"="checkpoints/Boomsday_toy_slot_attention.ckpt"
        conv_features = torch.clamp(x.flatten(start_dim=1),min = -10, max = 10)
        #object_scores = torch.sigmoid(0.000015 * self.object_score_marker(conv_features)) 
        score = torch.clamp( 0.001 * self.object_score_marker(conv_features), min = -10, max = 10)
        object_scores = torch.sigmoid(score) 
        object_features = self.object_feature_marker(conv_features)

        return img, logitmask, object_features,object_scores



class SceneGraphLevel(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_slots = 10
        in_dim = 2 + 128
        self.layer_embedding = nn.Parameter(torch.randn(10, in_dim))
        self.constuct_quarter = SlotAttention(num_slots,in_dim = in_dim,slot_dim = in_dim, iters = 5)

    def forward(self,inputs):
        in_scores = inputs["features"]
        in_features = inputs["scores"]
        B = in_scores.shape[0]

        if False:
            construct_features, construct_attn = self.connstruct_quarter(in_features)
            # [B,N,C]
        else:
            construct_features, construct_attn = in_features, in_scores

        proposal_features = self.layer_embedding.unsqueeze(0).repeat(B,1,1)

        match = torch.softmax(in_scores.unsqueeze(-1) * torch.einsum("bnc,bmc -> bnm",in_features, proposal_features)/math.sqrt(0.1), dim = -1)

        out_features = self.transfer(torch.einsum("bnc,bnm->bmc",construct_features, match))


        out_scores = torch.max(match).values
  


        return {"features":out_features,"scores":out_scores}

class SceneGraphNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = PSGNet(imsize = config.imsize, perception_size = config.perception_size)
        self.scene_graph_levels = nn.ModuleList([
            SceneGraphLevel(config)
        ])

    def forward(self, ims):
        primary_scene = self.backbone(ims)

        psg_features = to_dense_features(primary_scene)

        base_features = torch.cat([
            psg_features["features"],
            psg_features["centroids"],
        ],dim = -1)
        B = psg_features["features"].shape[0]
        P = psg_features["features"].shape[1]

        base_scene = {"scores":torch.ones(B,P,1),"features":base_features}
        abstract_scene = [base_scene]

        # [Construct the Scene Level]
        for merger in self.scene_graph_levels:
            construct_scene = merger(abstract_scene[-1])
            abstract_scene.append(construct_scene)

        primary_scene["abstract_scene"] = abstract_scene

        return primary_scene

        

class SceneTreeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = PSGNet(imsize = config.imsize, perception_size = config.perception_size)

        self.abstract_layers = nn.ModuleList([
            AbstractNet(64, 10)
        ])

    def load_backbone(self,backbone):self.backbone = backbone

    def forward(self, ims):
        primary_scene = self.backbone(ims)

        psg_features = to_dense_features(primary_scene)

        base_graph = psg_features[-1]

        working_graph = base_graph
        abstract_scene = [working_graph]
        for abstract_net in self.abstract_layers:
            working_graph = abstract_net(working_graph)
            abstract_scene.append(working_graph)
        
        primary_scene["abstract_scene"] = abstract_scene
        return primary_scene
