import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import math

class GraphConvolution(nn.Module):                            
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.unbalance = nn.Linear(in_features,in_features)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        B = inputs.shape[0]

        support = torch.matmul(self.unbalance(inputs), self.weight.unsqueeze(0).repeat(B,1,1))
        output = torch.matmul(adj, support)     
        if self.bias is not None:
            return output + self.bias         
        else:
            return output                      # (2708, 16)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LayerNorm([out_features]),
            nn.CELU(),
        )

    def forward(self, input):
        return self.net(input)

class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)

    def __getitem__(self,item):
        return self.net[item]

    def forward(self, input):
        return self.net(input)


class HierarchyBuilder(nn.Module):
    def __init__(self, config, output_slots, nu = 11):
        super().__init__()
        nu = 100 
        num_unary_predicates = nu
        num_binary_predicates = 0
        spatial_feature_dim = 0
        input_dim = num_unary_predicates + spatial_feature_dim + 1
        box = False
        self.num_unary_predicates = num_unary_predicates
        self.num_binary_predicates = num_binary_predicates
        self.graph_conv = GraphConvolution(input_dim,output_slots)
        if box:
            self.edge_predictor = FCBlock(128,3,input_dim * 2,1)
        else: self.edge_predictor = FCBlock(128,3,input_dim *2 ,1)
        self.dropout = nn.Dropout(0.01)
        self.attention_slots = nn.Parameter(torch.randn([1,output_slots,input_dim]))


    def forward(self, x, scores, executor):
        """
        input: 
            x: feature to agglomerate [B,N,D]
        """
        B, N, D = x.shape
        predicates = executor.concept_vocab

        if False:
            factored_features = [executor.entailment(
            x,executor.get_concept_embedding(predicate)
            ).unsqueeze(-1) for predicate in predicates]
            factored_features.append(scores)

            factored_features = torch.cat(factored_features, dim = -1)
        else:
            factored_features = torch.cat([x,scores], dim = -1)

        # [Perform Convolution on Factored States]
        GraphFactor = False
        if GraphFactor:
            adjs = self.edge_predictor(
            torch.cat([
                factored_features.unsqueeze(1).repeat(1,N,1,1),
                factored_features.unsqueeze(2).repeat(1,1,N,1),
            ], dim = -1)
            ).squeeze(-1)
            adjs = torch.sigmoid(adjs)
            adjs = self.dropout(adjs)

            adjs = torch.zeros([B, N, N])

            graph_conv_masks = self.graph_conv(factored_features, adjs).permute([0,2,1])
        else:
            #factored_features = F.normalize(factored_features)
            graph_conv_masks = torch.einsum("bnd,bmd->bmn",factored_features,\
                self.attention_slots.repeat(B,1,1)) 
        # [Build Connection Between Input Features and Conv Features]
        M = graph_conv_masks.shape[1]
        scale = 1/math.sqrt(D)
        #scale = 1
        gamma = 0.5

        graph_conv_masks = F.softmax(scale * (graph_conv_masks), dim = -1)

        g = graph_conv_masks
        #g = torch.sigmoid((graph_conv_masks - 0.5) * 10)
        #graph_conv_masks = graph_conv_masks / torch.sum(graph_conv_masks,dim =1,keepdim = True)
        g = g / g.sum( dim = 1, keepdim = True)
        #print(g,scores.squeeze(-1).repeat(1,M,1))

        #print(scores.squeeze(-1).unsqueeze(1).repeat(1,M,1).shape,g.shape)
        g = torch.min(scores.squeeze(-1).unsqueeze(1).repeat(1,M,1),g)

        #g = g /g.sum( dim = 1, keepdim = True)

        return g#graph_conv_masks #[B,10,7]

class DifferntialPool(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return  x