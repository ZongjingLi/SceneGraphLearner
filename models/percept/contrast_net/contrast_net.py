import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from types import SimpleNamespace
from .competition import *

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        self.args = args
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = args.RDNconfig
        """
        {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]
        """

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        if args.no_upsampling:
            self.out_dim = G0
        else:
            self.out_dim = args.n_colors
            # Up-sampling net
            if r == 2 or r == 3:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(r),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            elif r == 4:
                self.UPNet = nn.Sequential(*[
                    nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                    nn.PixelShuffle(2),
                    nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
                ])
            else:
                raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1

        if self.args.no_upsampling:
            return x
        else:
            return self.UPNet(x)

class ContrastNet__(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Conv. feature extractor to map pixels to feature vectors
        node_feat_size = config.node_feat_dim
        self.rdn = RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        self.part_num = config.object_num 
        self.mask_conv = nn.Conv2d(node_feat_size, self.part_num,5,1,2)\
#        RDN(SimpleNamespace(G0= self.part_num  ,RDNkSize=3,n_colors=node_feat_size,RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self, ims):
        """
        
        """
        device = self.device
        scale = 5
        conv_features = self.rdn(ims.permute(0,3,1,2)) # BxDxWxH
        conv_masks = torch.softmax(self.mask_conv(conv_features) * scale, dim = 1) # BxPxWxH

        B, W, H, P = conv_features.shape

        mask_weights = torch.einsum("bpwh->bp",conv_masks).unsqueeze(-1)
        base_features = torch.einsum("bdwh,bpwh->bpd",conv_features, conv_masks)
        base_features = base_features / mask_weights
        scores = conv_masks.max(-1).values.max(-1).values
        loss = 0
        loss += contrast_loss(base_features, conv_masks, conv_features)

        base_scene = [{"scores":scores,"features":base_features,"masks":conv_masks.permute(0,2,3,1),"match":False}]
        return {"abstract_scene":base_scene, "losses":loss}

class Id(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = nn.Parameter(torch.randn([1]))
    def forward(self, x):return x * self.k

class ContrastNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        conv_feat_dim = config.node_feat_dim
        self.convs = RDN(SimpleNamespace(G0=conv_feat_dim  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        #self.convs = Id()
        self.competition = Competition(num_masks = 7)

    def forward(self, ims):
        conv_feats = self.convs(ims.permute([0,3,1,2]) * 10).permute([0,2,3,1]) * 10
        #conv_feats = F.normalize(conv_feats, p = 2.0, dim = -1)
        masks, agents, alive, phenotypes, unharv = self.competition(conv_feats)

        masks = masks #* alive.unsqueeze(1).unsqueeze(1).squeeze(-1)
        #masks = torch.cat([masks, unharv], dim = -1)

        # raw features from the masked level
        scores = masks.max(1).values.max(1).values
        #masked_features = torch.einsum("bwhn,bwhd->bnd",masks,conv_feats)
        #node_features = masked_features / (torch.einsum("bwhn->bn",masks).unsqueeze(-1)) # [B,N,D]
        #node_features = torch.

        node_features = phenotypes
        #print(masks.permute(0,3,1,2).shape)
        base_scene = [{"scores":scores,"features":node_features,"masks":masks,"match":False}]
 
        return {"masks":masks,"abstract_scene":base_scene}

def contrast_loss(feats,conv_masks,conv_features, t = 0.07):
    B, P, W, H = conv_masks.shape
    mean_feat = feats.unsqueeze(2).unsqueeze(2).repeat(1,1,W,H,1) # BxPxWxHxD
    
    return 0

def contrastive_loss(feats, t=0.07):
    feats = F.normalize(feats, dim=2)  # B x K x C
    scores = torch.einsum('aid, bjd -> abij', feats, feats)
    scores = einops.rearrange(scores, 'a b i j -> (a i) (b j)')

    # positive logits: Nx1
    pos_idx = einops.repeat(torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    #pos_idx.fill_diagonal_(1)

    l_pos = torch.gather(scores, 1, pos_idx.nonzero()[:, 1].view(scores.size(0), -1))

    rand_idx = torch.randint(1, l_pos.size(1), (l_pos.size(0), 1), device=feats.device)
    l_pos = torch.gather(l_pos, 1, rand_idx)

    # negative logits: NxK
    neg_idx = einops.repeat(1-torch.eye(feats.size(1), dtype=torch.int, device=feats.device), 'i j -> (a i) (b j)', a=feats.size(0), b=feats.size(0))
    l_neg = torch.gather(scores, 1, neg_idx.nonzero()[:, 1].view(scores.size(0), -1))
    # logits: Nx(1+K)
    logits = torch.cat([l_pos, l_neg], dim=1)

    # apply temperature
    logits /= t

    # labels: positive key indicators
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=scores.device)
    return F.cross_entropy(logits, labels)