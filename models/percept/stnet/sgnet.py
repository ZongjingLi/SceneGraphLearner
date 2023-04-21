from .stnet import *

class SceneGraphLevel(nn.Module):
    def __init__(self, num_slots,config):
        super().__init__()
        iters = 10
        in_dim = config.object_dim
        self.layer_embedding = nn.Parameter(torch.randn(num_slots, in_dim))
        self.constuct_quarter = SlotAttention(num_slots,in_dim = in_dim,slot_dim = in_dim, iters = 5)
        self.hermit = nn.Linear(config.object_dim,in_dim)
        self.outward = nn.Linear(in_dim,in_dim, bias = False)
        self.propagator = GraphPropagation(num_iters = iters)

        node_feat_size = in_dim
        self.relation_encoder = RelationEncoder()
        self.graph_conv = GraphConv(node_feat_size , node_feat_size ,aggr = "mean") 

    def forward(self,inputs):
        in_features = inputs["features"]
        in_scores = inputs["scores"]
        B, N = in_scores.shape[0],in_scores.shape[1]


        raw_spatials = in_features[-2:]

        adjs = torch.tanh(0.1 * torch.linalg.norm(
         raw_spatials.unsqueeze(1).repeat(1,N,1,1) - 
         raw_spatials.unsqueeze(2).repeat(1,1,N,1), dim = -1)) 

        #edges = torch.sigmoid(adjs).int()


        if False:
            construct_features, construct_attn = self.connstruct_quarter(in_features)
            # [B,N,C]
        else:
            construct_features, construct_attn = in_features, in_scores
        construct_features[-2:] = 1 * construct_features[-2:]
        #construct_features = self.graph_conv(construct_features, edges)
        construct_features = self.propagator(construct_features,adjs)[-1]
        #construct_features = self.hermit(construct_features)

        proposal_features = self.layer_embedding.unsqueeze(0).repeat(B,1,1)

        match = torch.softmax(in_scores * torch.einsum("bnc,bmc -> bnm",in_features, proposal_features)/math.sqrt(0.1), dim = -1)

        out_features = torch.einsum("bnc,bnm->bmc",construct_features, match)
        #out_features = self.outward(out_features)

        out_scores = torch.max(match, dim = 1).values.unsqueeze(-1)

        in_masks = inputs["masks"]
        out_masks = torch.einsum("bwhm,bmn->bwhn",in_masks,match)



        return {"features":out_features,"scores":out_scores, "masks":out_masks, "match":match}

class SceneGraphNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = PSGNet(config.imsize, config.perception_size, config.object_dim - 2)
        self.scene_graph_levels = nn.ModuleList([
            SceneGraphLevel(5, config),
            #SceneGraphLevel(4, config)
        ])

    def forward(self, ims):
        # [PSGNet as the Backbone]
        B,W,H,C = ims.shape
        primary_scene = self.backbone(ims)
        psg_features = to_dense_features(primary_scene)[-1]

        base_features = torch.cat([
            psg_features["features"],
            psg_features["centroids"],
        ],dim = -1)
        B = psg_features["features"].shape[0]
        P = psg_features["features"].shape[1]

        # [Compute the Base Mask]
        clusters = primary_scene["clusters"]

        local_masks = []
        for i in range(len(clusters)):
            cluster_r = clusters[i][0];
            for cluster_j,batch_j in reversed(clusters[:i]):
                cluster_r = cluster_r[cluster_j].unsqueeze(0).reshape([B,W,H])

                local_masks.append(cluster_r)

        K = int(cluster_r.max()) + 1 # Cluster size
        local_masks = torch.zeros([B,W,H,K])
        
        for k in range(K):
            #local_masks[cluster_r] = 1
            local_masks[:,:,:,k] = torch.where(k == cluster_r,1,0)

        # [Construct the Base Level]
        base_scene = {"scores":torch.ones(B,P,1),"features":base_features,"masks":local_masks,"match":False}
        abstract_scene = [base_scene]

        # [Construct the Scene Level]
        for merger in self.scene_graph_levels:
            construct_scene = merger(abstract_scene[-1])
            abstract_scene.append(construct_scene)

        primary_scene["abstract_scene"] = abstract_scene

        return primary_scene

        

class RelationEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RelationEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )

    def forward(self, x):
        '''
        Args:
            x: [batch_size, n_relations, input_size]
        Returns:
            [batch_size, n_relations, output_size]
        '''
        B, N, D = x.size()
        x = self.model(x.view(B * N, D))
        return x.view(B, N, self.output_size)