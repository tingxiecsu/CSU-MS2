import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class SmilesModel(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(SmilesModel, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return out
        
class MlpBlock(nn.Module):
    def __init__(self, channels,hidden_channels):
        super(MlpBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, input):
        return self.block(input)


class CnnBlock_2(nn.Module):
    def __init__(self, channels):
        super(CnnBlock_2, self).__init__()
        self.layer = nn.Conv1d(1,
                               channels,
                               kernel_size=30,
                               stride=15,
                               padding=15)

    def forward(self, input):
        output_1 = self.layer(input.unsqueeze(1))
        return output_1


class MSModel_cnn(nn.Module):
    def __init__(self, output_channels, channels, input_channels=300):
        super(MSModel_cnn,self).__init__()
        hidden_channels = channels *21
        self.linears = nn.Sequential(
            CnnBlock_2(channels),
            nn.ReLU(),
            nn.Flatten(),
            MlpBlock(hidden_channels,hidden_channels),
            nn.ReLU(),
            MlpBlock(hidden_channels,hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_channels),
        )

    def forward(self,x):
        x = self.linears(x)
        return x

class ModelCLR(nn.Module):
    def __init__(self, num_layer, emb_dim, feat_dim, drop_ratio, pool,output_channels, channels, input_channels,embed_dim):
        super(ModelCLR, self).__init__()

        self.Smiles_model = SmilesModel(num_layer, emb_dim, feat_dim, drop_ratio, pool)
        #self.MS_model = MSModel( output_channels, channels=32, input_channels=300)
        self.MS_model = MSModel_cnn(output_channels, channels=32, input_channels=300)
        self.smi_proj = nn.Linear(output_channels, embed_dim)
        self.spec_proj = nn.Linear(output_channels, embed_dim)
    def smiles_encoder(self, xis):
        x = self.Smiles_model(xis)
        return x

    def ms_encoder(self, xls):
        out_emb = self.MS_model(xls)
        return out_emb

    def forward(self, xis, xls):
        zis = self.smiles_encoder(xis)
        zls = self.ms_encoder(xls)
        zis_feat=self.smi_proj(zis)
        zls_feat=self.spec_proj(zls)
        return zis_feat, zls_feat

#class ModelTransfer(nn.Module):