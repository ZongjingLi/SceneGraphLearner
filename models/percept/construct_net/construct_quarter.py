import torch
import torch.nn as nn

from torch_scatter         import scatter_mean

from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max
import torch_scatter

from .gcv import *
from .graph_propagation import *
from .node_extraction import *


def to_base(indices, edges):
    """
    input:
        indices: [K]
        edges: [N,2]
    outputs:
        base_indices: [L]
    """
    base_indices = []
    for i in range(edges.shape[0]): 
        if (edges[i][0] in indices): base_indices.append(i)
    return base_indices

def location_in_node(scene, node):
    return False

def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x

class SceneStructure:
    def __init__(self, graph, scores, from_base = None, base = None):
        self.graph = graph
        self.features = graph.x # [N, D]
        self.scores   = scores # [N, 1]
        self.edge_affinities = graph.edge_attr["weights"] #[N, N]
        self.from_base = from_base # [2,N]: [[1, 3],[1, 1]]
        self.base = base # Base Level Scene Structure
    
    def is_base(self): return self.from_base is None

    def locate_in(self, pos, node_indices):return 0

    def compute_masks(self, indices):
         # input: indices of nodes that need to compute mask
         # matrix form version.
        nodes = to_base(indices, self.from_base)
        if self.is_base():return self.scores[nodes]
        return self.base.compute_masks(nodes)

    def sparse_compute_masks(self, indices): 
        # input: indices of nodes that need to compute mask
        nodes = []
        for a in self.from_base.permute([1,0]):
            if a[1] in indices: nodes.append(a[0])
        if self.is_base():
            # this is the base level, just return the corresponding nodes
            return self.scores[nodes]
        return  self.base.compute_masks(nodes)



# prototype for the construct quarter
class ConstructQuarter(nn.Module):
    def __init__(self, in_feat_size, out_feat_size, k_nodes = 5, grid_size = 128 * 128):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # [Graph Convolution] for the input data
        self.graph_conv = GCNConv(in_feat_size, out_feat_size)
        self.location_itrs = k_nodes

        # [Affinity Decoder] softversion of graph constructer
        self.k_conv = GCNConv(in_feat_size, out_feat_size)
        self.q_conv = GCNConv(in_feat_size, out_feat_size)

        # [Graph Propagation] create the Graph Propgation Module
        self.graph_propagator = GraphPropagator(num_iters = 25,project=False,adj_thresh = 0.5)
        # GraphPropagator(num_iters = 7)

        # [Node Extraction]
        self.node_extractor = Competition(num_masks = k_nodes - 1)
        #NodeExtraction(k_nodes = k_nodes, grid_size = grid_size)
        
    def forward(self, node_features, node_edges = None, node_weights = None):
        # abstract the input graph data
        B,N,D = node_features.shape
        if node_edges is None:
            # need an extra step to build edges and decode affinities
            # [Abstract]
            edge_index = uniform_fully_connected(B, size = B)
            abstract_features = self.graph_conv(x, edge_index)

            # [Constructe Affinities]
            decode_ks = self.k_conv(x, edge_index)
            decode_qs = self.q_conv(x, edge_index)
            # build affinities
            weights = torch.cosine_similarity(decode_ks[:,edges[0,:],:] , decode_qs[:,edges[1,:],:], dim =-1)
            node__weights = softmax_max_norm(weights)

        # After this stage, {edge_index} ,{edge_weights} should be available
        # torch.Size([3, 16384, 64]) torch.Size([3, 2, 434724]) torch.Size([3, 434724])

        # [Propagate]
        # perform propagation over the continuous label on the graph 
        sparse_size = (B*N, B*N)
        Q = 128
        random_init_state = torch.randn([B,N,Q]) # random initialize labels : [B,N,D]
        # [B,N,D]   [B,2,n']    [B,n']
        rows = []; cols = []; ws = [];
        for b in range(B):rows.append(node_edges[b,0,:].long());cols.append(node_edges[b,1,:].long());ws.append(node_weights[b,:])
        rows = torch.cat(rows, dim = -1); cols = torch.cat(cols, dim = -1); ws = torch.cat(ws, dim = -1)
        adjs = SparseTensor(row=rows,\
                            col=cols,\
                            value=ws,
                            sparse_sizes=sparse_size)
        prop_features = self.graph_propagator(random_init_state, adjs) # [B,N,Q]
        W,H = 128,128

        platmap = prop_features.reshape([B,W,H,Q])
        # [Extract Nodes]
        # region competition and constuct the nodes at each level.
        masks, agents, alive, pheno, unharv = self.node_extractor(platmap)
        masks_extracted = torch.cat([masks, unharv], dim = -1).permute([0,3,1,2])

        #masks_extracted = self.node_extractor(prop_features)
        # B,M,N #["batch_node_masks"], node_outputs["batch_index"], node_outputs["sample_index"]
        B,N,_,_ = masks_extracted.shape
        node_features = torch.einsum("bnd,bmn->bmd",node_features, masks_extracted.reshape([B,N,W*H]))
        node_features = F.normalize(node_features, p=2, dim=1, eps=1e-12, out=None)

        node_scores = torch.max(torch.max(masks_extracted, dim = -1).values, dim = -1).values

        # node_features, node_masks, node_scores
        return node_features, masks_extracted, node_scores