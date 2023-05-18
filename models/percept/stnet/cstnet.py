from .stnet import *

class ControlPSGNet(torch.nn.Module):
    def __init__(self,imsize, perception_size, node_feat_size, struct = [1, 1]):

        super().__init__()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.imsize = imsize

        num_graph_layers = 2

        self.spatial_edges,self.spatial_coords = grid(imsize,imsize,device=device)
        
        # [Global Coords]
        self.global_coords = self.spatial_coords.float()

        self.spatial_edges = build_perception(imsize,perception_size,device = device)
        self.spatial_coords = self.spatial_coords.to(device).float() / imsize
        # Conv. feature extractor to map pixels to feature vectors
        self.rdn = RDN(SimpleNamespace(G0=node_feat_size  ,RDNkSize=3,n_colors=3,
                               RDNconfig=(4,3,16),scale=[2],no_upsampling=True))



        # Affinity modules: for now just one of P1 and P2 
        self.affinity_aggregations = torch.nn.ModuleList([])
        for s in struct:
            if s == 1: self.affinity_aggregations.append(P1AffinityAggregation())
            if s == 2: self.affinity_aggregations.append(P2AffinityAggregation())
        
        self.control_aggregations = torch.nn.ModuleList([])
        control = [10,4]
        for i in control:
            self.control_aggregations.append()
            

        # Node transforms: function applied on aggregated node vectors

        self.node_transforms = torch.nn.ModuleList([
            FCBlock(hidden_ch=100,
                    num_hidden_layers=3,
                    in_features =node_feat_size + 4,
                    out_features=node_feat_size,
                    outermost_linear=True) for _ in range(len(self.affinity_aggregations))
        ])

        # Graph convolutional layers to apply after each graph coarsening
        gcv = GraphConv(node_feat_size, node_feat_size)  
        self.graph_convs = torch.nn.ModuleList([
            GraphConv(node_feat_size , node_feat_size ,aggr = "mean")   for _ in range(len(self.affinity_aggregations))
        ])

        # Maps cluster vector to constant pixel color
        self.node_to_rgb  = FCBlock(hidden_ch=100,
                                    num_hidden_layers=3,
                                    in_features =20,
                                    out_features=3,
                                    outermost_linear=True)

        self.node_to_qtr_p1  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.node_to_qtr_p2  = FCBlock(100,3,node_feat_size,6 * 3,outermost_linear = True)
        self.gauge = nn.Linear(node_feat_size,node_feat_size)


    def forward(self,img,effective_mask = None):
        batch_size = img.shape[0]

        mask_shape = [batch_size,self.imsize,self.imsize,1]
        
        # [ Effective Mask Not Considered ]
        if effective_mask is None: effective_mask = torch.ones(mask_shape)


        # [Create the Local Coords]

        # Collect image features with rdn

        im_feats = self.rdn(img.permute(0,3,1,2))
        #im_feats = img.permute(0,3,1,2) 

        #coords_added_im_feats = torch.cat([
        #          self.spatial_coords.unsqueeze(0).repeat(im_feats.size(0),1,1),
        #          im_feats.flatten(2,3).permute(0,2,1)
        #                                  ],dim=2)
        coords_added_im_feats = im_feats.flatten(2,3).permute(0,2,1)

        ### Run image feature graph through affinity modules

        graph_in = Batch.from_data_list([Data(x,self.spatial_edges)
                                                for x in coords_added_im_feats])

        x, edge_index, batch = graph_in.x, graph_in.edge_index, graph_in.batch

        clusters, all_losses = [], [] # clusters just used for visualizations
        intermediates = [] # intermediate values

        ## Perform Affinity Calculation and Graph Clustering
        level_centroids = []
        level_moments = []
        level_batch = []
        
        for pool, conv, transf in zip(self.affinity_aggregations,
                                      self.graph_convs, self.node_transforms):
            batch_uncoarsened = batch

            x, edge_index, batch, cluster, losses = pool(x, edge_index, batch)
            level_batch.append(batch)

            clusters.append( (cluster, batch_uncoarsened) )
            for i,(cluster_r,_) in enumerate(clusters):
                for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
                device = self.device

                centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device),cluster_r.to(device),dim = 0)
                moments   = scatter_mean(self.spatial_coords.repeat(batch_size,1).to(device) ** 2,cluster_r.to(device),dim = 0 )
            
            x = x.to(self.device)
            edge_index = edge_index.to(self.device)
    

            # augument each node with explicit calculation of moment and centroids
            x = torch.cat([x,centroids,moments],dim = -1)
            x = transf(x)
            x = conv(x, edge_index)

            all_losses.append(losses)
            intermediates.append(x)

            level_centroids.append(centroids)
            level_moments.append(moments)
            
        # [joint Spatial features contruction with locality constraints]
        joint_spatial_features = []
        for i,(cluster_r,_) in enumerate(clusters):
            for cluster_j,_ in reversed(clusters[:i]):cluster_r = cluster_r[cluster_j]
            centroids = scatter_mean(self.spatial_coords.repeat(batch_size,1),cluster_r,dim = 0)

            joint_features = torch.cat([self.node_to_qtr_p2(intermediates[i]),centroids],-1)
            joint_spatial_features.append(joint_features)

        # [Add the reconstructions to the spatial edges for reconstruction loss]
        recons = [] # perform reconstruction over each layer composed
        for i,jsf in enumerate(joint_spatial_features):
            for cluster,_ in reversed(clusters[:i+1]):jsf = jsf[cluster]
            paint_by_numbers = to_dense_batch( 1.0 *  (\
                RenderQTR(self.spatial_coords.repeat(batch_size,1),jsf))\
                    ,graph_in.batch)[0]
            recons.append(paint_by_numbers)
            
        return {"recons":recons,"clusters":clusters,
        "losses":all_losses,
        "features":intermediates, 
        "centroids":level_centroids, 
        "moments":level_moments,
        "batch":level_batch}


class ControlSceneGraphNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = PSGNet(config.imsize, config.perception_size, config.object_dim - 2)

    def forward(self, ims):
        # [Calculate Mask at each level]

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


        primary_scene["abstract_scene"] = abstract_scene

        return primary_scene

