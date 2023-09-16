import torch
import torch.nn as nn
from types import SimpleNamespace
from .slot_attention import *
from .utils import *
from .convs import *
from models.nn.primary import *

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU(),
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

class GraphConvolution(nn.Module):

    def __init__(self, input_feature_num, output_feature_num, itrs = 5, add_bias=True, dtype=torch.float,
                 batch_normal=True):
        super().__init__()
        # shapes

        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.add_bias = add_bias
        self.batch_normal = batch_normal

        # params
        latent_dim = input_feature_num
        self.prim_net = nn.Identity() #FCBlock(128,2,input_feature_num, latent_dim)
        self.prop = nn.Linear(latent_dim, latent_dim)
        self.bias = nn.Parameter(torch.randn(latent_dim,dtype=dtype))
        self.transform = FCBlock(128,2,latent_dim, self.output_feature_num)
        
        self.sparse = True
        self.itrs = itrs
        #self.batch_norm = nn.BatchNorm1d(num_features = input_feature_num)
            
    def set_trainable(self, train=True):
        for param in self.parameters():
            param.requires_grad = train

    def forward(self, x, adj):
        """
        @param inp : adjacent: (batch, graph_num, graph_num) cat node_feature: (batch, graph_num, in_feature_num) -> (batch, graph_num, graph_num + in_feature_num)
        @return:
        """
        B, N, D = x.shape
        node_feature = x

        x = self.prim_net(node_feature)
        x = F.normalize(x, p = 2)
        #x = torch.nn.functional.normalize(x,p = 1.0, dim = -1, eps = 1e-5)
        for i in range(self.itrs):
            if self.sparse or isinstance(adj, torch.SparseTensor):
                x = x + torch.spmm(adj,x[0]).unsqueeze(0)
            else:
                #x = x + torch.matmul(adj,x[0])
                x = x + self.adj_split(x[0], adj)
            x = F.normalize(x,p = 2)

        x = self.transform(x)

        return x
    
    @staticmethod
    def adj_split(x, adj):
        # x: [N,D] adj:[N,N]
        if True:
            weights = torch.tanh(torch.einsum("nd,nd->nn"))
            weights = weights * adj
            x = x + torch.mm(weights,x)
        return x
    
class GraphConvolution_(nn.Module):
    def __init__(self, input_dim, output_dim, itrs = 3):
        super().__init__()
        hidden_dim = 128
        self.prim_encoder = nn.Linear(input_dim, hidden_dim)

        self.edge_propagators = nn.Linear(hidden_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.itrs = itrs

    def forward(self, x, adj):
        # x:[B,N,D] adj:[B,N,N]
        prop_features = self.prim_encoder(x)

        for i in range(self.itrs):
            if len(adj.shape) == 3:
                prop_features = torch.bmm(adj, prop_features)
                prop_features = self.edge_propagators(prop_features)
            else: 
                prop_features = torch.mm(adj, prop_features.squeeze(0)).unsqueeze(0)
                prop_features = self.edge_propagators(prop_features)
            
        outputs = self.classifier(prop_features)
        return outputs

class GNNSoftPooling(nn.Module):
    def __init__(self, input_feat_dim, output_node_num = 5, itrs = 1):
        super().__init__()
        self.assignment_net = GraphConvolution(input_feat_dim, output_node_num, itrs)
        self.feature_net =   GraphConvolution(input_feat_dim, input_feat_dim, itrs) 

    def forward(self, x, adj):
        B,N,D = x.shape
        if isinstance(adj, list):
            output_node_features = []
            output_new_adj = []
            output_s_matrix = []
            scale = 1.0
            for i in range(len(adj)):
                s_matrix = self.assignment_net(x[i:i+1], adj[i]) #[B,N,M]
                s_matrix = torch.softmax(s_matrix * scale , dim = 2)#.clamp(0.0+eps,1.0-eps)
            
                node_features = self.feature_net(x[i:i+1],adj[i]) #[B,N,D]
                node_features = torch.einsum("bnm,bnd->bmd",s_matrix,node_features) #[B,M,D]
                # [Calculate New Cluster Adjacency]

                adj[i] = adj[i]

                new_adj = torch.spmm(
                    torch.spmm(
                        s_matrix[0].permute(1,0),adj[i]
                        ),s_matrix[0])
                #new_adj = new_adj / new_adj.max()

                output_node_features.append(node_features)
                output_new_adj.append(new_adj)
                output_s_matrix.append(s_matrix)

            output_node_features = torch.cat(output_node_features, dim = 0)
            output_s_matrix = torch.cat(output_s_matrix, dim = 0)
        return output_node_features,output_new_adj,output_s_matrix

class SlotSoftPooing(nn.Module):
    def __init__(self, input_feat_dim, output_node_num = 10, itrs = 1):
        super().__init__()
        self.slot_attention = SlotAttention(output_node_num, input_feat_dim , input_feat_dim, 7)

    def forward(self, x , adj = None):
        # X: [B,N,D] adj: None
        
        inputs = x
        slot_features, att = self.slot_attention(inputs)

        output_node_features = slot_features
        M = output_node_features.shape[1]
        output_s_matrix = att.permute(0,2,1)
        output_new_adj = torch.ones(B,M,M)
        return output_node_features,output_new_adj,output_s_matrix

def get_fourier_feature(grid, term = 7):
    output_feature = []
    for k in range(term):
        output_feature.append(torch.sin(grid * (k + 1)))
        output_feature.append(torch.cos(grid * (k + 1)))
    output_feature = torch.cat(output_feature, dim = -1)
    return output_feature

class ObjectRender(nn.Module):
    def __init__(self,config, conv_feature_dim):
        super().__init__()
        channel_dim = config.channel
        spatial_dim = config.spatial_dim
        fourier_dim = config.fourier_dim

        self.conv_feature_dim = conv_feature_dim
        self.render_block  = FCBlock(128,3,conv_feature_dim + spatial_dim + spatial_dim + 2*spatial_dim*fourier_dim,channel_dim)

    def forward(self, latent, grid):
        B,N,D = latent.shape
        if len(grid.shape) == 4:
            # grid: [B,W,H,2]
            B, W, H, _ = grid.shape
            expand_latent = latent.unsqueeze(2).unsqueeze(2)
            expand_latent = expand_latent.repeat(1,1,W,H,1)
            grid = grid.unsqueeze(1)
            grid = grid.repeat(1,N,1,1,1)
        if len(grid.shape) == 3:
            # grid: [B,WH,2]
            B, WH, _ = grid.shape
            grid = grid.unsqueeze(1).repeat(1,N,1,1)
            expand_latent = latent.unsqueeze(2).repeat(1,1,WH,1)
        cat_feature = torch.cat([grid, expand_latent], dim = -1)
        return torch.sigmoid(self.render_block(cat_feature) * 3 )


class ValkyrNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.device
        # construct the grid domain connection
        self.imsize = config.imsize
        self.perception_size = config.perception_size
        # build the connection graph for the grid domain
        self.spatial_coords = grid(self.imsize,self.imsize,device=device)
        self.spatial_fourier_features = get_fourier_feature(self.spatial_coords, term = config.fourier_dim).to(device)
        self.spatial_edges =  build_perception(self.imsize,self.perception_size,device = device).to_dense().to(device)
        # [Grid Convs]
        conv_feature_dim = config.conv_feature_dim
        self.grid_convs = RDN(SimpleNamespace(G0=conv_feature_dim  ,RDNkSize=3,n_colors=3,RDNconfig=(4,3,16),scale=[2],no_upsampling=True))
        
        # [Diff Pool Construction]
        graph_pool = "GNN"
        hierarchy_nodes = config.hierarchy_construct 
        if graph_pool == "GNN":
            self.diff_pool = nn.ModuleList([
                GNNSoftPooling(input_feat_dim = conv_feature_dim+2,output_node_num = node_num ,itrs = config.itrs) for node_num in hierarchy_nodes
            ])
        if graph_pool == "Slot":
            self.diff_pool = nn.ModuleList([
                SlotSoftPooing(input_feat_dim = conv_feature_dim+2,output_node_num = node_num ) for node_num in hierarchy_nodes
            ])

        # [Render Fields]
        self.render_fields = nn.ModuleList([ObjectRender(config, conv_feature_dim) for _ in hierarchy_nodes])

        self.conv2object_feature = nn.Linear(conv_feature_dim + 2, config.object_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def forward(self, x, verbose = 0):
        outputs = {}
        B,W,H,C = x.shape # input shape
        device = self.device

        # [Grid Convolution] produce initial feature in the grid domain 
        grid_conv_feature = self.grid_convs(x.permute(0,3,1,2)).to(device).permute(0,2,3,1) 

        _,_,_,D = grid_conv_feature.shape
        coords_added_conv_feature = torch.cat(
            [grid_conv_feature, self.spatial_coords.unsqueeze(0).repeat(B,1,1,1).to(device)], dim = 3
        )
        if verbose:print("coords_added_conv_feature:{}x{}x{}x{}".format(*list(coords_added_conv_feature.shape) ))

        coords_added_conv_feature = coords_added_conv_feature.reshape(B,W*H,(D+2))
        coords_added_conv_feature = F.normalize(coords_added_conv_feature, dim = 2, p=1.0)
       
        # [DiffPool] each layer performs differentiable [Pn invariant] pooling 

        convs_features = []
        cluster_assignments = []
        curr_x = coords_added_conv_feature # base layer feature
  
        curr_edges = [self.spatial_edges for _ in range(B)] # base layer edges
        convs_features.append(curr_x)
        entropy_regular = 0.0 # initialize the entropy loss
        loc_loss = 0.0        # localization loss
        equi_loss = 0.0       # equillibrium loss
        scene_tree = {
            "x":[curr_x],
            "object_features":[self.conv2object_feature(curr_x)],
            "object_scores":[torch.ones(B,curr_x.shape[1]).to(self.device)],
            "connections":[],
            "edges":[self.spatial_edges]}
        outputs["masks"] = []
        outputs["poses"] = []

        layer_reconstructions = []
        layer_masks = [torch.ones(B,curr_x.shape[1]).to(self.device)]  # maintain a mask
        for i,graph_pool in enumerate(self.diff_pool):
            curr_x, curr_edges, assignment_matrix = graph_pool(curr_x, curr_edges)
            B,N,M = assignment_matrix.shape
            assignment_matrix = scene_tree["object_scores"][-1].unsqueeze(2).repeat(1,1,M) * assignment_matrix

            # previous level mask calculation
            prev_mask = layer_masks[-1]
            if len(prev_mask.shape) == 2:
                layer_mask = assignment_matrix #[BxNxWxHx1]
            else:layer_mask = torch.bmm(prev_mask,assignment_matrix)


            layer_masks.append(layer_mask)
                        
            #exist_prob = torch.max(assignment_matrix,dim = 1).values
            exist_prob = torch.ones(B, assignment_matrix.shape[-1]).to(device)

            # [Equivariance Loss]
            equis =assignment_matrix.unsqueeze(1).unsqueeze(-1)
            equi_loss += equillibrium_loss(equis) 
            
            # [Frobenius Term]

            for b in range(len(curr_edges)):
                equi_loss += 0.0 # frobenius_norm(scene_tree["edges"][-1][b], assignment_matrix[b])
            
            cluster_assignments.append(assignment_matrix)
            convs_features.append(curr_x)

            # [Scene Reconstruction]
            syn_grid = torch.cat([self.spatial_coords.to(device)\
                                  ,self.spatial_fourier_features.to(device)], dim = -1).unsqueeze(0).repeat(B,1,1,1)

            layer_recons = self.render_fields[i](
                curr_x,
                syn_grid
                )

            if verbose: print("reconstruction with shape: ", layer_recons.shape)
            layer_reconstructions.append(layer_recons)
            
            if verbose:print(assignment_matrix.max(),assignment_matrix.min(), curr_edges[0].shape, curr_edges[0].max(), curr_edges[0].min())
            
            # [Regular Entropy Term]
            
            layer_mask
            outputs["masks"].append(layer_mask)
            
            points = self.spatial_coords.unsqueeze(0).repeat(B,1,1,1).reshape(B,W*H,2)

            variance = spatial_variance(points, layer_mask.permute(0,2,1), norm_type="l2")
            loc_loss += 1.0 * variance.mean()

            # [Poses]
            # [B,N,K] [B,N,2]
            poses = torch.matmul(layer_mask.permute(0,2,1),points)/layer_mask.sum(1).unsqueeze(-1)
   
            outputs["poses"].append({"centers":poses,"vars":variance})

            entropy_regular += assignment_entropy(assignment_matrix)

            # load results to the scene tree
            scene_tree["x"].append(curr_x)
            scene_tree["object_features"].append(self.conv2object_feature(curr_x))
            scene_tree["object_scores"].append(exist_prob)
            scene_tree["connections"].append(assignment_matrix)
            scene_tree["edges"].append(curr_edges)

        # [Calculate Reconstruction at Each Layer]
        outputs["reconstructions"] = []

        reconstruction_loss = 0.0

        for i,recons in enumerate(layer_reconstructions):

            B,N,W,H,C = recons.shape

            exist_prob = scene_tree["object_scores"][i+1]\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,1,W,H,C) 

            mask = layer_masks[i+1].permute(0,2,1).reshape(B,N,W,H,1)

            recons = recons * mask * exist_prob
            
            #layer_recon_loss = torch.nn.functional.mse_loss(recons, x.unsqueeze(1).repeat(1,N,1,1,1))
            layer_recon_loss = torch.nn.functional.mse_loss(recons.sum(dim = 1), x)
            reconstruction_loss += layer_recon_loss
            outputs["reconstructions"].append(recons)


        # [Output the Scene Tree]
        outputs["scene_tree"] = scene_tree

        # [Add all the loss terms]
        #outputs["masks"] = layer_masks
        outputs["losses"] = {"entropy":entropy_regular*0,
        "reconstruction":reconstruction_loss,"equi":equi_loss*0,"localization":loc_loss*0}
        return outputs

def frobenius_norm(A, S):
    # A: [N,N] S:[N,M]
    coarse_connections = (torch.mm(S, S.permute(1,0)) - A).flatten()
    return torch.norm(coarse_connections,dim = -1)

def evaluate_pose(x, att):
    # x: BN3, att: BKN
    # ts: B3k1
    att = att.unsqueeze(1).unsqueeze(-1)
    x = x.permute(0,2,1)
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)
    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1
    return ts.permute(0,2,1,3).squeeze(-1)

def spatial_variance(x, att, norm_type="l2"):
    # att: BKN x: BN3
    x = x.permute(0,2,1)
    att = att.unsqueeze(1).unsqueeze(-1)
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
        vol = vol.norm(dim=1)
    elif norm_type == "l1":
        vol = vol.sum(dim=1)
    else:
        # vol, _ = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2).max(dim=1)
        raise NotImplementedError
    return vol

def assignment_entropy(s_matrix):
    # s_matrix: B,N,M
    EPS = 1e-6
    output_entropy = 0
    for b in range(s_matrix.shape[0]):
        for i in range(s_matrix.shape[1]):
            input_tensor = s_matrix[b][i:i+1,:].clamp(EPS, 1-EPS)

            lsm = nn.LogSoftmax(dim = -1)
            log_probs = lsm(input_tensor)
            probs = torch.exp(log_probs)
            p_log_p = log_probs * probs
            entropy = -p_log_p.mean()
            #print(entropy)
            output_entropy += entropy

    return output_entropy
    

def equillibrium_loss(att):
    pai = att.sum(dim=3, keepdim=True) # B1K11
    loss_att_amount = torch.var(pai.reshape(pai.shape[0], -1), dim=1).mean()
    return loss_att_amount

def build_perception(size,length,device):
    edges = [[],[]]
    for i in range(size):
        for j in range(size):
            # go for all the points on the grid
            coord = [i,j];loc = i * size + j

            for dx in range(-length,length+1):
                for dy in range(-length,length+1):
                    if i+dx < size and i+dx>=0 and j+dy<size and j+dy>=0:
                        if (i+dx) * size + (j + dy) != loc:
                            edges[0].append(loc)
                            edges[1].append( (i+dx) * size + (j + dy))
                            edges[0].append( (i+dx) * size + (j + dy))
                            edges[1].append(loc)
    outputs = torch.sparse_coo_tensor(edges, torch.ones(len(edges[0])), size = (size**2, size**2))
    return outputs.to(device)

def grid(width, height, device = "cuda:0" if torch.cuda.is_available() else "cpu"):
    x = torch.linspace(0,1,width)
    y = torch.linspace(0,1,height)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.cat([grid_x.unsqueeze(0),grid_y.unsqueeze(0)], dim = 0).permute(1,2,0)