import torch
import torch.nn as nn
from .loss import *
from .acne import *
from .decoder import *

def evaluate_pose(x, att):
    # x: B3N, att: B1KN1
    # ts: B3k1
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)

    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1
    return ts

class CSQModule(nn.Module):
    def __init__(self, config, num_slots):
        super().__init__()
        concept_dim = config.concept_dim
        self.num_slots = num_slots
        if config.conpcept_projection:
            self.feature2concept = nn.Linear(config.latent_dim, concept_dim)
    
    
    def forward(self,x):
        scores = 1
        node_features = 1
        masks = 1
        outputs = {"scores":scores,"features":node_features,"masks":masks,"match":False}
        return outputs

class CSQNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        concept_dim = config.concept_dim
        construct = ()
        self.base_encoder = AcneKpEncoder(config, indim = 4) # [Encoder]

        gc_dim = config.acne_dim 
        self.decoder = KpDecoder(self.config.acne_num_g, gc_dim,
            self.config.num_pts, self.config)                   # [Decoder]
        self.csq_modules = [CSQModule(config, num_slots) for num_slots in construct]
        if config.concept_projection:
            self.feature2concept = nn.Linear(config.latent_dim, concept_dim)
        self.scaling = 1.0

    def forward(self, inputs):
        enc_in = inputs['point_cloud'] * self.scaling 
        query_points = inputs['coords'] * self.scaling 
        

        enc_in = torch.cat([enc_in, inputs['rgb']], 2)[...,None].permute(0,2,1,3)

        f_att = self.base_encoder(inputs['point_cloud'] * self.scaling , return_att=True)
        base_feature, attention = f_att
        pos = evaluate_pose(base_feature, attention)

        base_feature = base_feature.squeeze(3)
        attention = attention.squeeze(1)
        attention = attention.squeeze(3)

        scene_construct = {"scores":1,"features":1,"masks":1,"raw_features":0,"match":False}

        # [Construct the Hierarchical Representation]
        for csqnet in self.csq_modules:
            csqnet(scene_construct)

        losses = {"reconstruction":1.0,"localization":0.1}
        outputs = {"loss":losses,"occ":[torch.tensor(1.0)],"color":[torch.tensor(0.0)]}
        return outputs