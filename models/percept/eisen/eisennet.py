import torch
import torch.nn as nn

from .propagation import *
from .competition import *

def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x

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
        affinity_res=[128, 128]
        kq_dim = 32
        supervision_level = 3
        conv_feat_dim = 64
        output_dim = conv_feat_dim
        subsample_affinity=True,
        eval_full_affinity=False,
        local_window_size=25
        num_affinity_samples=1024
        propagation_iters=25
        propagation_affinity_thresh=0.7
        num_masks=32
        num_competition_rounds=3
        supervision_level=3
        # Feature Decoder

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

    
    def forward(self, ims):
        pass
