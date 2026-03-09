import torch
import math
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from g4satbench.models.graph_modules import SAGE, GCN, GNN, Pooling, GIN
from g4satbench.models.mlp import MLP


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True, GNN_type='gnn', pooling='max'):
        super(Encoder, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        if GNN_type == 'sage':
            self.gnn = SAGE(input_dim * 2, hidden_dim, hidden_dim, layer_num, dropout)
        elif GNN_type == 'gcn':
            self.gnn = GCN(input_dim * 2, hidden_dim, hidden_dim, layer_num, dropout)
        elif GNN_type == 'gnn':
            self.gnn = GNN(input_dim * 2, hidden_dim, hidden_dim, layer_num, dropout)
        elif GNN_type == 'gin':
            self.gnn = GIN(input_dim=3, output_dim=1, hidden_dim=hidden_dim, num_layers=layer_num,
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum")
        else:
            raise NotImplementedError
        if GNN_type != 'gin':
            self.init_emb = nn.Parameter(torch.randn(1, input_dim) * math.sqrt(2 / input_dim))
            self.k_encoder = MLP(2, 1, hidden_dim, input_dim, 'relu')
            self.pooling = Pooling(pooling_method=pooling)
            self.projector = nn.Linear(hidden_dim, output_dim)
        else:
            self.init_emb = nn.Parameter(torch.randn(1, 2))
            self.k_encoder = MLP(2, 1, 1, input_dim, 'relu')
            self.pooling = Pooling(pooling_method=pooling)
            self.projector = nn.Linear(1, output_dim)

    def forward(self, data):
        init_emb = torch.repeat_interleave(self.init_emb, data.num_nodes, dim=0)

        init_k = torch.zeros(data.num_nodes, 1).to(data.ptr.device)
        for i in range(data.ptr.size(0)):
            if i == 0:
                continue
            init_k[data.ptr[i-1]:data.ptr[i]] = data.k[i-1]

        k_emb = self.k_encoder(init_k)
        init_emb = torch.cat([init_emb, k_emb], dim=1)

        x = self.gnn(data, init_emb)
        x = self.pooling(x, data.ptr)
        x = self.projector(x)
        return x


class GraphModel(nn.Module):
    def __init__(self, opts):
        super(GraphModel, self).__init__()
        self.opts = opts
        self.encoder = Encoder(self.opts.dim, self.opts.dim, self.opts.dim, self.opts.graph_layer_num,
                               self.opts.dropout, self.opts.gragh_gnn_type, self.opts.pooling)

        self.mlp = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)

    def forward(self, data):
        rep = self.encoder(data)
        pred = torch.sigmoid(self.mlp(rep).reshape(-1))
        return rep, pred
