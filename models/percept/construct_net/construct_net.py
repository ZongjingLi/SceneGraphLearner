from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

import torch_scatter
import torch_geometric

from torch_scatter         import scatter_mean

from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max

from types import SimpleNamespace

from .construct_quarter import *
from .graph_propagation import *
from .projection import *
from .projection import *
from .convs import *
from .gcv import *

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j
            
            for r in range(1):
                random_long_range = torch.randint(128, (1,2) )[0]
                #edges[0].append(random_long_range[0] // size)
                #edges[1].append(random_long_range[1] % size)
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        if (i+dx) * size + (j + dy) != loc:
                            edges[0].append(loc)
                            edges[1].append( (i+dx) * size + (j + dy))
    return torch.tensor(edges).to(device)

def sample_indices(batch, size, k_samples):
    max_index = size
    sample_index = []
    batch_index = []
    for b in range(batch):
        for k in range(k_samples): 
            sample_index.append(np.random.randint(max_index*b, max_index*(b+1)))
            batch_index.append(b)
    return sample_index, batch_index

def uniform_fully_connected(batch_size = 3, size = 30):
    assert size % batch_size == 0, print("full size cannot be batchify")
    full_edges = []
    for b in range(batch_size):
        for i in range(size // batch_size):
            for j in range(size // batch_size):full_edges.append([b * i,b * j])
    full_edges = torch.tensor(full_edges).t()
    return full_edges

# [Scene Structure]
import math
def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x


    
class ConstructNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.device
        # construct the grid domain connection
        self.imsize = config.imsize
        self.perception_size = config.perception_size
        # build the connection graph for the grid domain
        spatial_edges, self.spatial_coords = grid(self.imsize,self.imsize,device=device)
        self.spatial_edges =  build_perception(self.imsize,self.perception_size,device = device)
    
        node_feat_size = config.node_feat_dim
        # [Grid Convolution]
        self.grid_convs =RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        
        # [Affinity Decoder]
        kq_dim = node_feat_size
        latent_dim = node_feat_size
        norm_fn = "batch"
        kernel_size = 3
        downsample = False
        self.k_convs = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))
        self.q_convs = nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))

        # [Construct Quarters]
        construct_config = (128*128, 6)
        self.construct_quarters = nn.ModuleList(
            [ConstructQuarter(node_feat_size, node_feat_size, construct_config[i+1], construct_config[i]) for i in range(len(construct_config) - 1)]
        )

        self.verbose = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def forward(self, ims):
        # flatten the image-feature and add it with the coordinate information
        B, W, H, C = ims.shape
        im_feats = self.grid_convs(ims.permute([0,3,1,2])) # [B,D,W,H]

        # [Image Grid Convolution]
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1); # 【B, N, D】


        # [Affinity Decoder](base)
        edges = self.spatial_edges
        decode_ks = self.k_convs(im_feats).flatten(2,3).permute(0,2,1)
        decode_qs = self.q_convs(im_feats).flatten(2,3).permute(0,2,1)

        weights = torch.cosine_similarity(decode_ks[:,edges[0,:],:] , decode_qs[:,edges[1,:],:], dim =-1)# * (C ** -0.5) 
        weights = softmax_max_norm(weights)
        edges = self.spatial_edges.unsqueeze(0).repeat(B,1,1)
        #print("weights_shape:{} max:{} min:{}".format(weights.shape, weights.max(), weights.min()))
        # [Base Graph] construct the initial spatial-augumented graph input        

        # [Construct Quarter] 
        node_features = coords_added_im_feats; P = node_features.shape[1]; 
        construct_scene = [{"scores":torch.ones(B,P,1).to(self.device),"features":node_features,"masks":1.0,"match":False}]
        # create the abstracted graph at each level and construct the scene parse tree
        for i,construct_quarter in enumerate(self.construct_quarters):
            # input to construct quarter: edges[B,2,N] weights [B,N]
            if i == 0: node_features, node_masks, node_scores = construct_quarter(node_features, edges, weights)
            else: node_features, node_masks, node_scores = construct_quarter(node_features)
            # output of construct quarter: nodes(feature): [B,M,D] masks: [B,M,d^2]
            # scores of each mask can be calculated as s = max(Mask)
            abstract_scene = {"scores":node_scores,"features":node_features,"masks":node_masks,"match":False}
            construct_scene.append(abstract_scene)
    

        return {"gt_im":ims, "abstract_scene":construct_scene}
