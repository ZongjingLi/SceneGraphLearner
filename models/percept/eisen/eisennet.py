import torch
import torch.nn as nn

from .propagation import *

def softmax_max_norm(x):
    x = x.softmax(-1)
    x = x / torch.max(x, dim=-1, keepdim=True)[0].clamp(min=1e-12)# .detach()
    return x

class EisenNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        kq_dim = 32
        supervision_level = 3
        conv_feat_dim = 64
        output_dim = conv_feat_dim
        subsample_affinity=True,
        eval_full_affinity=False,
        local_window_size=25
        num_affinity_samples=1024,
        propagation_iters=25,
        propagation_affinity_thresh=0.7,
        num_masks=32,
        num_competition_rounds=3,
        supervision_level=3,
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
            self.register_buffer(buffer_name, utils.generate_local_indices(img_size=[H, W], K=local_window_size).cuda(), persistent=False)

        # [Propagation]
        self.propagation = GraphPropagation(num_iters=propagation_iters, adj_thresh=propagation_affinity_thresh)

        # [Competition]
        self.competition = Competition(num_masks=num_masks, num_competition_rounds=num_competition_rounds)

    
    def forward(self, ims):
        pass

if __name__ == "__main__":
    from config import *
    eisennet = EisenNet(config)