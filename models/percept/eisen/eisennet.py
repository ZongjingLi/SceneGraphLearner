from types import SimpleNamespace
import torch
import torch.nn as nn

from .convnet     import *
from .propagation import *
from .competition import *

def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x
def local_to_sparse_global_affinity(local_adj, sample_inds, activated=None, sparse_transpose=False):
    """
    Convert local adjacency matrix of shape [B, N, K] to [B, N, N]
    :param local_adj: [B, N, K]
    :param size: [H, W], with H * W = N
    :return: global_adj [B, N, N]
    """

    B, N, K = list(local_adj.shape)

    if sample_inds is None:
        return local_adj

    assert sample_inds.shape[0] == 3
    local_node_inds = sample_inds[2] # [B, N, K]

    batch_inds = torch.arange(B).reshape([B, 1]).to(local_node_inds)
    node_inds = torch.arange(N).reshape([1, N]).to(local_node_inds)
    row_inds = (batch_inds * N + node_inds).reshape(B * N, 1).expand(-1, K).flatten()  # [BNK]

    col_inds = local_node_inds.flatten()  # [BNK]
    valid = col_inds < N

    col_offset = (batch_inds * N).reshape(B, 1, 1).expand(-1, N, -1).expand(-1, -1, K).flatten() # [BNK]
    col_inds += col_offset
    value = local_adj.flatten()

    if activated is not None:
        activated = activated.reshape(B, N, 1).expand(-1, -1, K).bool()
        valid = torch.logical_and(valid, activated.flatten())

    if sparse_transpose:
        global_adj = SparseTensor(row=col_inds[valid], col=row_inds[valid],
                                  value=value[valid], sparse_sizes=[B*N, B*N])
    else:
        raise ValueError('Current KP implementation assumes tranposed affinities')

    return global_adj

def downsample_tensor(x, stride):
    # x should have shape [B, C, H, W]
    if stride == 1:
        return x
    B, C, H, W = x.shape
    x = F.unfold(x, kernel_size=1, stride=stride)  # [B, C, H / stride * W / stride]
    return x.reshape([B, C, int(H / stride), int(W / stride)])

def gather_tensor(tensor, sample_inds, invalid=0.):
    # tensor is of shape [B, N, D]
    # sample_inds is of shape [2, B, T, K] or [3, B, T, K]
    # where the last column of the 1st dimension are the sample indices

    _, N, D = tensor.shape
    dim, B, T, K = sample_inds.shape


    if dim == 2:
        if sample_inds[-1].max() == N:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, 1, D], device=tensor.device)], dim=1)

        indices = sample_inds[-1].view(B, T * K).unsqueeze(-1).expand(-1, -1, D)
        output = torch.gather(tensor, 1, indices).view([B, T, K, D])
    elif dim == 3:
        if sample_inds[-1].max() == D:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, N, 1], device=tensor.device)], dim=2)
            D = D + 1
        elif sample_inds[1].max() == N:
            # special case: indices where idx == N is assigned zero
            tensor = torch.cat([tensor, invalid * torch.ones([B, 1, D], device=tensor.device)], dim=1)
            N = N + 1

        tensor = tensor.view(B, N * D)
        node_indices = sample_inds[1].view(B, T * K)
        sample_indices = sample_inds[2].view(B, T * K)
        indices = node_indices * D + sample_indices
        # print('in gather tensor: ', indices.max(), tensor.shape)
        output = torch.gather(tensor, 1, indices).view([B, T, K])
    else:
        raise ValueError
    return output

def generate_local_indices(img_size, K, padding='constant'):
    H, W = img_size
    indice_maps = torch.arange(H * W).reshape([1, 1, H, W]).float()

    # symmetric padding
    assert K % 2 == 1  # assert K is odd
    half_K = int((K - 1) / 2)

    assert padding in ['reflection', 'constant'], "unsupported padding mode"
    if padding == 'reflection':
        pad_fn = torch.nn.ReflectionPad2d(half_K)
    else:
        pad_fn = torch.nn.ConstantPad2d(half_K, H * W)

    indice_maps = pad_fn(indice_maps)
    local_inds = F.unfold(indice_maps, kernel_size=K, stride=1)  # [B, C * K * k, H, W]
    local_inds = local_inds.permute(0, 2, 1)
    return local_inds

class EisenNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        affinity_res=[128, 128]
        self.affinity_res = affinity_res
        kq_dim = 32
        supervision_level = 1
        conv_feat_dim = 64
        output_dim = conv_feat_dim
        subsample_affinity=True,
        eval_full_affinity=False,
        local_window_size=25
        self.local_window_size = local_window_size
        self.num_affinity_samples=1024
        propagation_iters= 55
        propagation_affinity_thresh=0.7
        num_masks= 6 - 1
        num_competition_rounds=5
        self.supervision_level = supervision_level
        # [Feature Decoder]
        self.grid_conv =  self.rdn = RDN(SimpleNamespace(G0=conv_feat_dim  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        # [Affinity decoder]
        self.feat_conv = nn.Conv2d(output_dim, kq_dim, kernel_size=1, bias=True, padding='same')
        self.key_proj = nn.Linear(kq_dim, kq_dim)
        self.query_proj = nn.Linear(kq_dim, kq_dim)

        # [Affinity sampling]
        self.sample_affinity = subsample_affinity and (not (eval_full_affinity and (not self.training)))
        for level in range(supervision_level):
            stride = 2 ** level
            H, W = affinity_res[0]//stride, affinity_res[1]//stride
            buffer_name = f'local_indices_{H}_{W}'
            self.register_buffer(buffer_name, generate_local_indices(img_size=[H, W], K=local_window_size).to(device), persistent=False)

        # [Propagation]
        self.propagation = GraphPropagation(num_iters=propagation_iters, adj_thresh=propagation_affinity_thresh)

        # [Competition]
        self.competition = Competition(num_masks=num_masks, num_competition_rounds=num_competition_rounds)

        self.verbose = False
    
    def forward(self, ims):
        if self.verbose: print("ims: BxWxHxC: {}x{}x{}x{}".format(*list(ims.shape)))
        conv_features = self.grid_conv(ims.permute(0,3,1,2)).permute(0,2,3,1) * 10
        
        if self.verbose: print("conv_features: BxWxHxD: {}x{}x{}x{}".format(*list(conv_features.shape)))

        kq_features = self.feat_conv(conv_features.permute(0,3,1,2)).permute(0,2,3,1)
        key = self.key_proj(kq_features).permute(0, 3, 1, 2) # [B, C, H, W]
        query = self.query_proj(kq_features).permute(0, 3, 1, 2) # [B, C, H, W]
        B, C, H, W = key.shape

        if self.verbose: print("\nkey:BxKxWxH {}x{}x{}x{} query:BxQxWxH {}x{}x{}x{}".format(*list(key.shape),*list(query.shape)))

        sample_inds = None
        affinity_list = []
        # Compute affinity loss at multiple scales
        for level in range(self.supervision_level if self.training else 1):
            stride = 2 ** level

            # [Sampling affinities]
            if self.sample_affinity:
                sample_inds = self.generate_affinity_sample_indices(size=[B, H//stride, W//stride])

            # [Compute affinity logits]
            affinity_logits = self.compute_affinity_logits(
                key=downsample_tensor(key, stride),
                query=downsample_tensor(query, stride),
                sample_inds=sample_inds
            ) * (C ** -0.5)

            affinity_list.append(affinity_logits)
        segments,alive,unharv = self.compute_segments(affinity_list[0], sample_inds)

        masks = torch.cat([segments, unharv], dim = -1)

        masked_features = torch.einsum("bwhn,bwhd->bnd",masks,conv_features)
        node_features = masked_features / (torch.einsum("bwhn->bn",masks).unsqueeze(-1))
        scores = torch.cat([alive, unharv.max(1).values.max(1).values.unsqueeze(-1)],dim=-2)
        scores = torch.min(scores,masks.max(1).values.max(1).values.unsqueeze(-1))
        
        # [Base Scene]
        base_scene = [
            {"scores":scores,"features":node_features,"masks":masks,"match":False}
            ]

        if self.verbose: print("segments: BxWxHxN {}x{}x{}x{}",segments.shape)
        return {"masks":segments, "abstract_scene":base_scene}

    def compute_segments(self, logits, sample_inds, hidden_dim=32, run_cc=True, min_cc_area=20):
        B, N, K = logits.shape

        # [Initialize hidden states]
        h0 = torch.FloatTensor(B, N, hidden_dim).normal_().softmax(-1).to(self.device) # h0

        # [Process affinities]
        adj = softmax_max_norm(logits) # normalize affinities
        # Convert affinity matrix to sparse tensor for memory efficiency, if subsample_affinity = True
        adj = local_to_sparse_global_affinity(adj, sample_inds, sparse_transpose=True) # sparse affinity matrix

        # [Graph propagation]
        plateau_map_list = self.propagation(h0.detach(), adj.detach())
        plateau_map = plateau_map_list[-1].reshape([B, self.affinity_res[0], self.affinity_res[1], hidden_dim])

        # [Competition]
        masks, agents, alive, phenotypes, unharvested = self.competition(plateau_map)

        return masks, alive, unharvested#{"abstract_scene":base_scene}

    def compute_affinity_logits(self, key, query, sample_inds):
        B, C, H, W = key.shape
        key = key.reshape([B, C, H * W]).permute(0, 2, 1)      # [B, N, C]
        query = query.reshape([B, C, H * W]).permute(0, 2, 1)  # [B, N, C]

        if self.sample_affinity: # subsample affinity
            gathered_query = gather_tensor(query, sample_inds[[0, 1], ...])
            gathered_key = gather_tensor(key, sample_inds[[0, 2], ...])
            logits = (gathered_query * gathered_key).sum(-1)  # [B, N, K]
        else: # full affinity
            logits = torch.matmul(query, key.permute(0, 2, 1))
        return logits
    def generate_affinity_sample_indices(self, size):
        B, H, W = size
        S = self.num_affinity_samples
        K = self.local_window_size
        # local_indices and local_masks below are stored in the buffers
        # so that we don't have to repeat the same computation at every iteration
        local_inds = getattr(self, f'local_indices_{H}_{W}').expand(B, -1, -1)  # tile local indices in batch dimension
        device = local_inds.device

        if K ** 2 <= S:
            # sample random global indices
            rand_global_inds = torch.randint(H * W, [B, H*W, S-K**2], device=device)
            sample_inds = torch.cat([local_inds, rand_global_inds], -1)
        else:
            sample_inds = local_inds

        # create gather indices
        sample_inds = sample_inds.reshape([1, B, H*W, S])
        batch_inds = torch.arange(B, device=device).reshape([1, B, 1, 1]).expand(-1, -1, H*W, S)
        node_inds = torch.arange(H*W, device=device).reshape([1, 1, H*W, 1]).expand(-1, B, -1, S)
        sample_inds = torch.cat([batch_inds, node_inds, sample_inds], 0).long()  # [3, B, N, S]

        return sample_inds