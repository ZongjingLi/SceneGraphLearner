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
# prototype for the construct quarter
class ConstructQuarter(nn.Module):
    def __init__(self, in_feat_size, out_feat_size, k_nodes = 5, grid_size = 128 * 128):
        super().__init__()
        # [Graph Convolution] for the input data
        self.graph_conv = GCNConv(in_feat_size, out_feat_size)
        self.location_itrs = k_nodes

        # [Affinity Decoder] softversion of graph constructer
        self.k_conv = GCNConv(in_feat_size, out_feat_size)
        self.q_conv = GCNConv(in_feat_size, out_feat_size)

        # [Graph Propagation] create the Graph Propgation Module
        self.graph_propagator = GraphPropagator(num_iters = 75,project=True,adj_thresh = 0.5)
        # GraphPropagator(num_iters = 7)

        # [Node Extraction]
        self.node_extractor = NodeExtraction(k_nodes = k_nodes, grid_size = grid_size)
        
    def forward(self, scene, from_base = False, verbose = False):
        # abstract the input graph data
        if from_base:
            # do not compute affinity graph
            input_graph = scene.graph
            x = input_graph.x
            edge_index = input_graph.edge_index
            edge_weights = input_graph.edge_attr["weights"]
            batch_size = input_graph.batch.max() + 1
            abstract_features = x
        else:
            # need an extra step to build edges and decode affinities
            # [Abstract]
            input_graph = scene.graph
            x = input_graph.x
            batch_size = input_graph.batch.max() + 1
            edge_index = uniform_fully_connected(batch_size, size = scene.graph.x.shape[0])
            abstract_features = self.graph_conv(x, edge_index)

            # [Constructe Affinities]
            decode_ks = self.k_conv(x, edge_index)
            decode_qs = self.q_conv(x, edge_index)
            # build affinities
            if verbose:print(decode_ks.shape)
            weights = torch.cosine_similarity(
            decode_ks[edge_index[0,:],:],decode_qs[edge_index[1,:],:],
             dim = -1)
            weights = softmax_max_norm(weights)
            if verbose:
                print("start the Graph Convolution")
                print(abstract_features.shape)        
            edge_weights = weights
        
        # After this stage, {edge_index} ,{edge_weights} should be available

        # [Propagate]
        # perform propagation over the continuous label on the graph
        random_init_state = torch.randn(x.shape) # random initialize labels
        if 0:
            prop_features = self.graph_propagator(random_init_state, edge_index, edge_weights)
        else:
            B = batch_size; N = random_init_state.shape[0]
            random_init_state = random_init_state.unsqueeze(0)
            sparse_size = (N,N)
            adjs = SparseTensor(row=edge_index[0,:].long(),col=edge_index[1,:].long(),value=edge_weights,sparse_sizes=sparse_size)
            prop_features = self.graph_propagator(random_init_state, adjs)[0,:,:]
        if verbose:
            print("start the Graph Propagation")
            print("prop_features:",prop_features.shape)
        
        # [Extract Nodes]
        # region competition and constuct the nodes at each level.
        node_outputs = self.node_extractor(prop_features, scene)
        masks_extracted, masks_batch, raw_features, sample_index = node_outputs #["batch_node_masks"], node_outputs["batch_index"], node_outputs["sample_index"]
        masks_extracted = torch.tensor(masks_extracted)
        if verbose:
            print(abstract_features.shape, masks_extracted.shape)
        node_features = torch.einsum("nd,mn->md",abstract_features, masks_extracted)
        node_scores = torch.max(masks_extracted, dim = -1).values
        if verbose:
            print("start the Node Extraction")
            print("  masks_extracted:", masks_extracted.shape)
            print("  node_features:", node_features.shape)
            print("  node_scores:", node_scores.shape)
        # TODO: a more complicated node extraction
    
        output_graph = Batch.from_data_list([Data(node_features, edge_attr={"weights":None})])
        output_graph.batch = torch.tensor(masks_batch)

        # [Build Abstracted Scene] 
        abstract_scene = SceneStructure(output_graph, node_scores, False, scene)

        return abstract_scene, masks_extracted, raw_features, sample_index