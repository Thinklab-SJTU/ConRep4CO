import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F

import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.utils import to_dense_batch


import dgl
import dgl.function as fn
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import MaxPooling


class MLP_GIN(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        h = F.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class GIN(nn.Module):
    def __init__(self, input_dim=3, output_dim=1, hidden_dim=256, num_layers=5,
                 graph_level_output=0, learn_eps=False, dropout=0.,
                 aggregator_type="sum"):
        super().__init__()

        self.inp_embedding = nn.Embedding(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.inp_transform = nn.Identity()

        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        assert aggregator_type in ["sum", "mean", "max"]
        for layer in range(num_layers - 1):  # excluding the input layer
            mlp = MLP_GIN(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, aggregator_type=aggregator_type, learn_eps=learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.output_dim = output_dim
        self.graph_level_output = graph_level_output
        # linear functions for graph poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            self.linear_prediction.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim+graph_level_output))
            )
        self.readout = nn.Sequential(
            nn.Linear(num_layers * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim+graph_level_output)
        )
        self.drop = nn.Dropout(dropout)
        self.pool = MaxPooling()

    def forward(self, g, state, reward_exp=None):
        assert reward_exp is None

        h = self.inp_embedding(state)
        h = self.inp_transform(h)
        # list of hidden representation at each layer
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = self.readout(torch.cat(hidden_rep, dim=-1))

        return score_over_layer





# SAGE
class SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True):
        super(SAGE, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)

        self.conv_hidden = tg.nn.SAGEConv(hidden_dim, hidden_dim)

        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data, x):
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):

            # x = self.conv_hidden[i](x, edge_index)
            x = self.conv_hidden(x, edge_index)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


# GCN
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True):
        super(GCN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)

        self.conv_hidden = tg.nn.GCNConv(hidden_dim, hidden_dim)

        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data, x):
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):

            # x = self.conv_hidden[i](x, edge_index)
            x = self.conv_hidden(x, edge_index)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


# GNN, GraphConv
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True):
        super(GNN, self).__init__()
        self.layer_num = layer_num
        self.dropout = dropout
        self.conv_first = tg.nn.GraphConv(input_dim, hidden_dim)

        self.conv_hidden = tg.nn.GraphConv(hidden_dim, hidden_dim)

        self.conv_out = tg.nn.GraphConv(hidden_dim, output_dim)

    def forward(self, data, x):
        edge_index = data.edge_index
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):

            # x = self.conv_hidden[i](x, edge_index)
            x = self.conv_hidden(x, edge_index)

            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


# Pooling and mergeing to acquire graph-level representation with the inputed node-level representation
class Pooling(torch.nn.Module):
    def __init__(self, pooling_method='mean'):
        super(Pooling, self).__init__()
        self.pooling_method = pooling_method

    def forward(self, x, pointer):
        # create batch from pointer
        batch = torch.zeros(pointer[-1], dtype=torch.long, device=x.device)
        for i in range(len(pointer) - 1):
            batch[pointer[i]:pointer[i + 1]] = i
        if self.pooling_method == 'mean':
            x = tg.nn.global_mean_pool(x, batch)
        elif self.pooling_method == 'max':
            x = tg.nn.global_max_pool(x, batch)
        elif self.pooling_method == 'sum':
            x = tg.nn.global_add_pool(x, batch)
        return x


class GraphGPS(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num=2, dropout=True):
        super(GraphGPS, self).__init__()
        self.local_gnn_type = 'GCN'
        self.global_model_type = 'Transformer'
        self.num_heads = 4
        self.layer_num = 4

        for i in range(self.layer_num):
            setattr(self, f'layer_{i}', GPSLayer(
                dim_h=hidden_dim,
                local_gnn_type=self.local_gnn_type,
                global_model_type=self.global_model_type,
                num_heads=self.num_heads,
                act='relu',
                dropout=dropout,
                attn_dropout=0.0,
                layer_norm=False,
                batch_norm=True,
                bigbird_cfg=None,
                log_attn_weights=False
            ))

    def forward(self, data, x):
        for i in range(self.layer_num):
            x = getattr(self, f'layer_{i}')(data, x)
        return x


class GPSLayer(nn.Module):
        """Local MPNN + full graph attention x-former layer.
        """
        def __init__(self, dim_h,
                     local_gnn_type, global_model_type, num_heads, act='relu',
                     pna_degrees=None, equivstable_pe=False, dropout=0.0,
                     attn_dropout=0.0, layer_norm=False, batch_norm=True,
                     bigbird_cfg=None, log_attn_weights=False):
            super().__init__()

            self.dim_h = dim_h
            self.num_heads = num_heads
            self.attn_dropout = attn_dropout
            self.layer_norm = layer_norm
            self.batch_norm = batch_norm
            self.equivstable_pe = equivstable_pe
            self.activation = register.act_dict[act]

            self.log_attn_weights = log_attn_weights
            if log_attn_weights and global_model_type not in ['Transformer',
                                                              'BiasedTransformer']:
                raise NotImplementedError(
                    f"Logging of attention weights is not supported "
                    f"for '{global_model_type}' global attention model."
                )

            # Local message-passing model.
            self.local_gnn_with_edge_attr = True
            if local_gnn_type == 'None':
                self.local_model = None

            # MPNNs without edge attributes support.
            elif local_gnn_type == "GCN":
                self.local_gnn_with_edge_attr = False
                self.local_model = pygnn.GCNConv(dim_h, dim_h)
            else:
                raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
            self.local_gnn_type = local_gnn_type

            # Global attention transformer-style model.
            if global_model_type == 'None':
                self.self_attn = None
            elif global_model_type in ['Transformer', 'BiasedTransformer']:
                self.self_attn = torch.nn.MultiheadAttention(
                    dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
                # self.global_model = torch.nn.TransformerEncoderLayer(
                #     d_model=dim_h, nhead=num_heads,
                #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
                #     layer_norm_eps=1e-5, batch_first=True)
            else:
                raise ValueError(f"Unsupported global x-former model: "
                                 f"{global_model_type}")
            self.global_model_type = global_model_type

            if self.layer_norm and self.batch_norm:
                raise ValueError("Cannot apply two types of normalization together")

            # Normalization for MPNN and Self-Attention representations.
            if self.layer_norm:
                self.norm1_local = pygnn.norm.LayerNorm(dim_h)
                self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
                # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
                # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
                # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
                # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
            if self.batch_norm:
                self.norm1_local = nn.BatchNorm1d(dim_h)
                self.norm1_attn = nn.BatchNorm1d(dim_h)
            self.dropout_local = nn.Dropout(dropout)
            self.dropout_attn = nn.Dropout(dropout)

            # Feed Forward block.
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
            self.act_fn_ff = self.activation()
            if self.layer_norm:
                self.norm2 = pygnn.norm.LayerNorm(dim_h)
                # self.norm2 = pygnn.norm.GraphNorm(dim_h)
                # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
            if self.batch_norm:
                self.norm2 = nn.BatchNorm1d(dim_h)
            self.ff_dropout1 = nn.Dropout(dropout)
            self.ff_dropout2 = nn.Dropout(dropout)

        def forward(self, batch, x):
            h = x
            h_in1 = h  # for first residual connection

            h_out_list = []
            # Local MPNN with edge attributes.
            if self.local_model is not None:
                self.local_model: pygnn.conv.MessagePassing  # Typing hint.

                h_local = self.local_model(h, batch.edge_index)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.
                if self.layer_norm:
                    h_local = self.norm1_local(h_local, batch.batch)
                if self.batch_norm:
                    h_local = self.norm1_local(h_local)
                h_out_list.append(h_local)

            # Multi-head attention.
            if self.self_attn is not None:
                h_dense, mask = to_dense_batch(h, batch.batch)
                if self.global_model_type == 'Transformer':
                    h_attn = self._sa_block(h_dense, None, ~mask)[mask]
                else:
                    raise RuntimeError(f"Unexpected {self.global_model_type}")

                h_attn = self.dropout_attn(h_attn)
                h_attn = h_in1 + h_attn  # Residual connection.
                if self.layer_norm:
                    h_attn = self.norm1_attn(h_attn, batch.batch)
                if self.batch_norm:
                    h_attn = self.norm1_attn(h_attn)
                h_out_list.append(h_attn)

            # Combine local and global outputs.
            # h = torch.cat(h_out_list, dim=-1)
            h = sum(h_out_list)

            # Feed Forward block.
            h = h + self._ff_block(h)
            if self.layer_norm:
                h = self.norm2(h, batch.batch)
            if self.batch_norm:
                h = self.norm2(h)

            return h

        def _sa_block(self, x, attn_mask, key_padding_mask):
            """Self-attention block.
            """
            if not self.log_attn_weights:
                x = self.self_attn(x, x, x,
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)[0]
            else:
                # Requires PyTorch v1.11+ to support `average_attn_weights=False`
                # option to return attention weights of individual heads.
                x, A = self.self_attn(x, x, x,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask,
                                      need_weights=True,
                                      average_attn_weights=False)
                self.attn_weights = A.detach().cpu()
            return x
