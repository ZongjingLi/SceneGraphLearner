# write some util functions

import numpy as np

import torch
import torch.nn as nn

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

# [Node Extraction]

class NodeExtraction(nn.Module):
    def __init__(self, k_nodes = 5, grid_size = 128*128):
        super().__init__()
        self.k_nodes = k_nodes
        self.grid_size = grid_size

    def forward(self,state, scene):
        """
        input: scene structure with
            x: node features [N, D]
            from_base: connection to the lower level
            locate_in(loc, node) -> bool
        output: at most k nodes from the 
        """
        node_features = state #[N,D]
        batch_size = scene.graph.batch.max() + 1

        #print(batch_size)
        sample_index, index_batch = sample_indices(batch_size, size = self.grid_size, k_samples = self.k_nodes)
        
        batch_node_mask = []
        for b in range(batch_size):
            #print(b)
            sample_features = node_features[sample_index[self.k_nodes*b: self.k_nodes*(b+1)]]
            #print("location_feature_heads: ", sample_features.shape)
            #print("node_features:", node_features.shape)
            #masks = torch.einsum("nd,md->nm", sample_features, node_features)
            #print(sample_features.shape, node_features.shape)
            print(torch.matmul(sample_features, node_features.permute(1,0)).max(),torch.matmul(sample_features, node_features.permute(1,0)).min())
            masks = torch.sigmoid(
                (torch.matmul(sample_features, node_features.permute(1,0)) - 0.25) / 0.02
                ) 
            # mask out components out of the batch
            for b_ in range(batch_size):
                if b != b:masks[self.k_nodes*b_: self.k_nodes*(b_+1), :] *= 0.0
            print("max:{} min:{}".format(masks.max(), masks.min()))
            batch_node_mask.append(masks)
            #print("masks:",masks.shape)
        batch_node_mask = torch.cat(batch_node_mask, dim = 1)
        return batch_node_mask, index_batch
