import torch
import torch.nn as nn

def spatial_variance(x, att, norm_type="l2"):
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)
    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1

    x_centered = x[:, :, None] - ts # B3KN
    x_centered = x_centered.permute(0, 2, 3, 1) # BKN3
    att = att.squeeze(1) # BKN1
    cov = torch.matmul(
        x_centered.transpose(3, 2), att * x_centered) # BK33
    
    # l2 norm
    vol = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2) # BK
    if norm_type == "l2":
        vol = vol.norm(dim=1).mean()
    elif norm_type == "l1":
        vol = vol.sum(dim=1).mean()
    else:
        # vol, _ = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2).max(dim=1)
        raise NotImplementedError
    return vol