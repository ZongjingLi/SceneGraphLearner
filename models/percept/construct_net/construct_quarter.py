import torch
import torch.nn as nn

from torch_scatter         import scatter_mean

from torch_geometric.nn    import max_pool_x, GraphConv
from torch_geometric.data  import Data,Batch
from torch_geometric.utils import grid, to_dense_batch
from torch_scatter import scatter_mean,scatter_max

from gcv import *
from graph_propagation import *
from node_extraction import *

# prototype for the construct quarter
class ConstructQuarter(nn.Module):
    def __init__(self, in_feat_size, out_feat_size, k_nodes = 5):
        super().__init__()
        # [Graph Convolution] for the input data
        self.graph_conv = GCNConv(in_feat_size, out_feat_size)
        self.location_itrs = k_nodes

        # [Affinity Decoder] softversion of graph constructer
        self.k_conv = GCNConv(in_feat_size, out_feat_size)
        self.q_conv = GCNConv(in_feat_size, out_feat_size)

        # [Graph Propagation] create the Graph Propgation Module
        self.graph_propagator = GraphPropagator(num_iters = 25)
        # GraphPropagator(num_iters = 7)

        # [Node Extraction]
        self.node_extractor = NodeExtraction(k_nodes = k_nodes, grid_size = k_nodes)
        
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

            weights = torch_scatter.scatter_softmax(weights, edge_index[1,:]) # softmax((Wfi).(Wfj))
            #weights = torch.sigmoid(weights)
            weights = weights / torch.max(weights,edge_index[1,:])
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
            #adjs = torch.zeros([B,N,N])
            #adjs[0,edge_index[0,:], edge_index[1,:]] = edge_weights
            random_init_state = random_init_state.unsqueeze(0)
            sparse_size = (N,N)
            adjs = SparseTensor(row=edge_index[0,:].long(),col=edge_index[1,:].long(),value=edge_weights,sparse_sizes=sparse_size)
            prop_features = self.graph_propagator(random_init_state, adjs)[0,:,:]
        if verbose:
            print("start the Graph Propagation")
            print("prop_features:",prop_features.shape)
        
        # [Extract Nodes]
        # region competition and constuct the nodes at each level.
        masks_extracted, masks_batch = self.node_extractor(prop_features, scene)
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
        print("abstract_node_features:",node_features.shape)
    
        output_graph = Batch.from_data_list([Data(node_features, edge_attr={"weights":None})])
        output_graph.batch = torch.tensor(masks_batch)

        # [Build Abstracted Scene] 
        abstract_scene = SceneStructure(output_graph, node_scores, False, scene)

        return abstract_scene, masks_extracted