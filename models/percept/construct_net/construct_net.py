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
            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
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



class ConstructNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.device
        # construct the grid domain connection
        self.imsize = config.imsize
        self.perception_size = 5#config.perception_size
        # build the connection graph for the grid domain
        spatial_edges, self.spatial_coords = grid(self.imsize,self.imsize,device=device)
        self.spatial_edges =  build_perception(self.imsize,self.perception_size,device = device)
    
        node_feat_size = 64
        # [Grid Convolution]
        self.grid_convs =RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        
        # [Affinity Decoder]
        kq_dim = node_feat_size
        latent_dim = node_feat_size
        norm_fn = "batch"
        kernel_size = 3
        downsample = False
        self.k_convs = \
        nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))
        self.q_convs = \
        nn.Sequential(
            ResidualBlock(kq_dim, latent_dim, norm_fn, kernel_size=kernel_size, bias=False, stride=1, residual=True, downsample=downsample),
            nn.Conv2d(latent_dim, kq_dim, kernel_size=1, bias=True, padding='same'))

        # [Construct Quarters]
        construct_config = (10,5)
        self.construct_quarters = nn.ModuleList(
            [ConstructQuarter(node_feat_size, node_feat_size, k) for k in construct_config]
        )

        self.verbose = 0

    def forward(self, ims):
        # flatten the image-feature and add it with the coordinate information
        B, W, H, C = ims.shape
        if self.verbose:
            print("input_image:\n  {}x{}x{}x{} #BxWxHxC".format(*list(ims.shape)))
        im_feats = self.grid_convs(ims.permute([0,3,1,2])) # [B,D,W,H]
        #im_feats = ims.permute([0,3,1,2])

        # [Image Grid Convolution]
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1) # 【B, N, D】
        weights = torch.ones(self.spatial_edges.shape[1]) # []

        # [Affinity Decoder](base)
        edges = self.spatial_edges
        decode_ks = self.k_convs(im_feats).flatten(2,3).permute(0,2,1)
        decode_qs = self.q_convs(im_feats).flatten(2,3).permute(0,2,1)

        weights = torch.einsum("bnd,bnd->bn",
            decode_ks[:,edges[0,:],:],decode_qs[:,edges[1,:],:],
            )
        #print(weights.max(), weights.min())
        #weights = torch_scatter.scatter_softmax(weights, edges[1,:]) # softmax((Wfi).(Wfj))
        weights = torch.sigmoid((weights))
        #weights = weights / torch.max(weights,edges[1,:])
        # TODO: scatter normalize the graph weight

        # [Base Graph] construct the initial spatial-augumented graph input        
        graph_in = Batch.from_data_list([
            Data(coords_added_im_feats[i], self.spatial_edges, edge_attr = {"weights":weights[i]})
                                                for i in range(B)])

        if self.verbose:
            print("input_graph:\n x: {}x{} #NxD \n  batch: {} bn:{}\n  edge_indices: {}x{}\n  edge_weight: {}".format(
                *list(graph_in.x.shape), *list(graph_in.batch.shape), graph_in.batch.max()+1,
                *list(graph_in.edge_index.shape), *list(graph_in.edge_attr["weights"].shape)
                ))
            print("  weight specs: max:{} min:{}".format(graph_in.edge_attr["weights"].max(), graph_in.edge_attr["weights"].min()))

        # [Construct Quarter] 
        # create the abstracted graph at each level and construct the scene parse tree
        input_graph = graph_in
        construct_counter = 0
        # base level scene structure
        base_scene_structure = SceneStructure(input_graph,\
            scores = torch.ones(weights.shape), from_base = None, base = None)
        curr_scene = base_scene_structure
        from_base = True
        level_masks = []
        if self.verbose:
            print("scene construction::\n")
        for construct_quarter in self.construct_quarters:
            construct_counter +=1
            if self.verbose:
                print("construct quarter {}:".format(construct_counter))
            curr_scene, masks = construct_quarter(curr_scene, from_base)
            from_base = False
            level_masks.append(masks)
            # load the abstract graph information
            if self.verbose:
                print("")
        return {"gt_im":ims, "masks":level_masks}