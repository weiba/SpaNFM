import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import argparse
from entmax import entmax15
from torch_geometric.nn import GCNConv

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(adj, x) + self.bias
        return x


class LayerNorm(nn.Module):
    def __init__(self, feature_size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature_size))
        self.b_2 = nn.Parameter(torch.ones(feature_size))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1,keepdim=True)
        return  self.a_2 * (x - mean) / (std +self.eps) + self.b_2


class HieraFormer(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim,add_self=False, normalize_embedding=False, dropout=0.0, bias=True,IsMulti=True, mul_num=3):
        super(HieraFormer, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=self.dropout)
        self.normalize_embedding = normalize_embedding
        self.h = mul_num  #head numbers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        # self.softmax = nn.Softmax(dim=-1)
        self.IsMulti = IsMulti
        self.d_k = int(input_dim/self.h)
        self.dropout_x = nn.Dropout(p=0.1)
        if IsMulti:
            self.linears = nn.ModuleList()
            for i in range(3):
                Mul_outdim = int(input_dim/mul_num)*mul_num
                self.linear = nn.Linear(input_dim, Mul_outdim)
                self.linears.append(self.linear)
                self.norm =LayerNorm(Mul_outdim)
        else:
            self.linears = nn.ModuleList()
            for i in range(3):
                if i == 2 or i == 3:
                    self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
                else:
                    self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
                self.linears.append(self.linear)
                self.norm =LayerNorm(self.embedding_dim)

        if bias:
            self.bias = nn.Parameter(torch.rand(output_dim).cuda())
            nn.init.uniform_(self.bias, -0.1, 0.1)
        else:
            self.bias = None

    def attention(self,query, key, value, dropout=None):
        d_k = query.size(-1)
        energy = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(d_k))
        "sparse attention"
        attention = entmax15(energy, dim=-1)

        if dropout is not None:
            p_attn = self.dropout_x(attention)

        return torch.matmul(p_attn, value)
    def forward(self, x):
        B, _, _ = x.size()
        x = x.transpose(1, 2)
        x1 =x
        if self.IsMulti:
            q, k, v = [l(x_a).view(B, -1, self.h, self.d_k).transpose(1, 2) for l, x_a in zip(self.linears,(x,x,x))]
            x = self.attention(q, k, v, dropout=self.dropout)
            x = x.transpose(1,2).contiguous().view(B, -1, self.h*self.d_k)
        else:
            q, k, v = [l(x_a) for l, x_a in zip(self.linears, (x, x, x))]
            x = self.attention(q, k, v, dropout=self.dropout)
        x = self.dropout_x(x)
        x_norm = x1 + self.norm(x)
        return x_norm


class model_hierar(nn.Module):
    def __init__(self, args,concat=True,output_channels=2):
        super(model_hierar, self).__init__()
        self.args = args
        self.num_pooling = args.num_pooling
        embedding_dim = args.embedding_dim
        assign_ratio = args.assign_ratio
        assign_ratio_1 = args.assign_ratio_1
        partroi = args.partroi
        self.num_pooling = args.num_pooling
        num_pooling = args.num_pooling
        self.mult_num = args.mult_num
        self.dropout = args.dropout
        max_num_nodes = partroi
        self.node = max_num_nodes
        self.concat = concat
        add_self = not concat
        channel = 512

        self.input_channel = channel
        assign_input_dim = channel
        inter_channel = int(max_num_nodes * assign_ratio)

        inter_channel_1 = 128
        self.conv1 = nn.Conv1d(int(self.input_channel / self.mult_num) * self.mult_num, inter_channel, 1)
        self.conv2 = nn.Conv1d(int(self.input_channel / self.mult_num) * self.mult_num, inter_channel_1, 1)
        self.bn1 = nn.BatchNorm1d(inter_channel)
        self.bn2 = nn.BatchNorm1d(inter_channel_1)
        self.bias = True
        self.gcn_512 = GraphConvolution(1000, 512)

        assign_dims = []

        self.conv_first_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            if i == 0:
                self.pred_input_dim = inter_channel_1
            else:
                self.pred_input_dim = int(inter_channel/self.mult_num)*self.mult_num
            conv_first2 = self.build_hiera_layers(self.pred_input_dim,  embedding_dim, add_self, normalize=True, dropout=self.dropout,IsMulti=True, mul_num=self.mult_num)
            self.conv_first_after_pool.append(conv_first2)

        self.conv_first = self.build_hiera_layers(channel, embedding_dim, add_self, normalize=True, dropout=self.dropout,IsMulti=True,mul_num=self.mult_num)
        self.assign_conv_first_modules = nn.ModuleList()

        # self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first = self.build_hiera_layers(assign_input_dim,  assign_dim, add_self,normalize=True,dropout=self.dropout,IsMulti=False)
            if i == 0:
                assign_input_dim = inter_channel
            else:
                assign_input_dim = int(inter_channel/self.mult_num)*self.mult_num
            assign_dim = int(assign_dim * assign_ratio_1)
            self.assign_conv_first_modules.append(assign_conv_first)


        # self.softmax = nn.Softmax(dim=-1)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.bn =True
        
    def build_hiera_layers(self, input_dim,  embedding_dim,  add_self, normalize=False, dropout=0.0, mul_num=8, IsMulti=True):
        conv_first = HieraFormer(input_dim=input_dim, output_dim=self.input_channel, embedding_dim=embedding_dim, add_self=add_self,
                                      normalize_embedding=normalize, bias=self.bias,dropout=dropout, IsMulti=IsMulti, mul_num=mul_num)

        return conv_first
    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, conv_first, embedding_mask=None):
        x = x.permute(0, 2, 1)
        x_tensor = conv_first(x)
        if embedding_mask is not None:
            x_tensor = x * embedding_mask
        return x_tensor

    def edge2mat(self,num_node):
        self_link = [(i, i) for i in range(num_node)]
        A = np.zeros((num_node, num_node))
        for i, j in self_link:
            A[j, i] = 1
        print(A)
        print(A.shape)
        return A
    def forward(self, x, adj):
        x = self.gcn_512(x, adj)

        hierarchical_tensor = self.gcn_forward(x, self.conv_first)
    
        if self.num_pooling == 0:
            hierarchical_tensor = self.dp1(F.relu(self.bn1(self.conv1(hierarchical_tensor.permute(0, 2, 1))))).permute(0, 2, 1)
        else:
            for i in range(self.num_pooling):
                x = self.dp1(F.relu(self.bn1(self.conv1(x.permute(0, 2, 1))))).permute(0, 2, 1)
                self.NodeAssign_tensor = self.gcn_forward(x, self.assign_conv_first_modules[i])
                x = torch.matmul(torch.transpose(self.NodeAssign_tensor, 1, 2), hierarchical_tensor)
                if i == 0:
                    x = self.dp1(F.relu(self.bn2(self.conv2(x.permute(0, 2, 1))))).permute(0, 2, 1)

                hierarchical_tensor = self.gcn_forward(x, self.conv_first_after_pool[i])
    
        x = hierarchical_tensor

        return x, self.NodeAssign_tensor