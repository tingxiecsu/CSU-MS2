import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
import nn_utils as nn_utils
num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 4
num_hybrid_type = 8
num_valence_tag = 8
num_degree_tag = 5

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3
num_bond_configuration = 6
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
        #self.edge_embedding3 = nn.Embedding(num_bond_configuration, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        #nn.init.xavier_uniform_(self.edge_embedding3.weight.data)

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
        self.x_embedding3 = nn.Embedding(num_hybrid_type, emb_dim)
        self.x_embedding4 = nn.Embedding(num_valence_tag, emb_dim)
        self.x_embedding5 = nn.Embedding(num_degree_tag, emb_dim)
        
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        nn.init.xavier_uniform_(self.x_embedding3.weight.data)
        nn.init.xavier_uniform_(self.x_embedding4.weight.data)
        nn.init.xavier_uniform_(self.x_embedding5.weight.data)

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

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1]) + self.x_embedding3(x[:,2]) + self.x_embedding4(x[:,3]) + self.x_embedding5(x[:,4])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        '''h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)'''
        
        return h

class FourierEmbedder(nn.Module):
    """Embed a set of mz float values using frequencies"""

    def __init__(self, spec_embed_dim, logmin=2.5, logmax=3.3):
        super().__init__()
        self.d = spec_embed_dim
        self.logmin = logmin
        self.logmax = logmax

        lambda_min = np.power(10, -logmin)
        lambda_max = np.power(10, logmax)
        index = torch.arange(np.ceil(self.d / 2))
        exp = torch.pow(lambda_max / lambda_min, (2 * index) / (self.d - 2))
        freqs = 2 * np.pi * (lambda_min * exp) ** (-1)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # Turn off requires grad for freqs
        self.freqs.requires_grad = False

    def forward(self, mz: torch.FloatTensor):
        """forward

        Args:
            mz: FloatTensor of shape (batch_size, mz values)

        Returns:
            FloatTensor of shape (batch_size, peak len, mz )
        """
        freq_input = torch.einsum("bi,j->bij", mz, self.freqs)
        embedded = torch.cat([torch.sin(freq_input), torch.cos(freq_input)], -1)
        embedded = embedded[:, :, : self.d]
        return embedded

class MSModel(nn.Module):
    def __init__(self, spec_embed_dim,dropout,layers):
        super(MSModel,self).__init__()
        self.mz_embedder = FourierEmbedder(spec_embed_dim)
        self.input_compress = nn.Linear(spec_embed_dim+1, spec_embed_dim)
        peak_attn_layer = nn_utils.TransformerEncoderLayer(
           d_model=spec_embed_dim,
           nhead=8,
           dim_feedforward=spec_embed_dim * 4,
           dropout=dropout,
           additive_attn=False,
           pairwise_featurization=False)
        self.peak_attn_layers = nn_utils.get_clones(peak_attn_layer,layers)
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(spec_embed_dim, spec_embed_dim)
        
    def forward(self,mzs,intens,num_peaks):
        embedded_mz = self.mz_embedder(mzs)
        cat_vec = [embedded_mz, intens[:, :, None]]
        peak_tensor = torch.cat(cat_vec, -1)
        peak_tensor = self.input_compress(peak_tensor)
        peak_dim = peak_tensor.shape[1]
        peaks_aranged = torch.arange(peak_dim).to(mzs.device)

        # batch x num peaks
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])

        # Transpose to peaks x batch x features
        peak_tensor = peak_tensor.transpose(0, 1)
        for peak_attn_layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = peak_attn_layer(
                peak_tensor,
                src_key_padding_mask=attn_mask,
            )

        peak_tensor = peak_tensor.transpose(0, 1)

        # Get only the class token
        #h0 = peak_tensor[:, 0, :]

        #output = self.output_layer(h0)
        
        '''pooled_embeddings = self.pooling_layer(peak_tensor.permute(0, 2, 1)).squeeze(dim=-1)
        output = self.output_layer(pooled_embeddings)'''
        return peak_tensor,attn_mask

class ESA_SMILES(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)
        
    def forward(self, hidden_states,data_batch):
        B = data_batch.max().item() + 1  # batch_num
        node_counts = torch.bincount(data_batch)  # node_num
        N = node_counts.max().item()  # max_node_num
        C = hidden_states.shape[1]  # feat_dim
        result = torch.zeros((B, N, C)).to(hidden_states.device)
        for i in range(B):
            indices = torch.where(data_batch == i)[0]
            result[i, :len(indices), :] = hidden_states[indices]
        attention_mask = (result != 0).any(dim=-1).float() 
        logits = self.ln_f(result) # (B, N, C)
        cap_embes = self.linear(logits) # Q
        features_in = self.linear1(cap_embes) # M
        mask = attention_mask.unsqueeze(-1) # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4) # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        smi_feature = torch.sum(attn * cap_embes, dim=1) # (B, C)
        return smi_feature

class ESA_SPEC(nn.Module):
    def __init__(self, feature_dim, out_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, out_dim)
        
    def forward(self, hidden_states,attention_mask):
        logits = self.ln_f(hidden_states) # (B, N, C)
        cap_embes = self.linear(logits) # Q
        features_in = self.linear1(cap_embes) # M
        mask = attention_mask.unsqueeze(-1) # (B, N, 1)
        features_in = features_in.masked_fill(mask == 0, -1e4) # (B, N, C)
        features_k_softmax = nn.Softmax(dim=1)(features_in)
        attn = features_k_softmax.masked_fill(mask == 0, 0)
        spec_feature = torch.sum(attn * cap_embes, dim=1) # (B, C)
        return spec_feature

class ModelCLR(nn.Module):
    def __init__(self, num_layer, emb_dim, feat_dim, drop_ratio, pool,spec_embed_dim,dropout,layers,embed_dim):
        super(ModelCLR, self).__init__()

        self.Smiles_model = SmilesModel(num_layer, emb_dim, feat_dim, drop_ratio, pool)
        self.MS_model = MSModel(spec_embed_dim,dropout,layers)
        self.smi_esa = ESA_SMILES(emb_dim, embed_dim)
        self.spec_esa = ESA_SPEC(spec_embed_dim, embed_dim)
        self.smi_proj = nn.Linear(embed_dim, embed_dim)
        self.spec_proj = nn.Linear(embed_dim, embed_dim)

    def smiles_encoder(self, xis):
        x = self.Smiles_model(xis)
        return x

    def ms_encoder(self, mzs,intens,num_peaks):
        out_emb = self.MS_model(mzs,intens,num_peaks)
        return out_emb

    def forward(self, xis, mzs,intens,num_peaks):
        zis = self.smiles_encoder(xis)
        zls,attn_mask = self.ms_encoder(mzs,intens,num_peaks)
        zis_feat=self.smi_esa(zis,xis.batch)
        zls_feat=self.spec_esa(zls,attn_mask)
        zis_feat=self.smi_proj(zis_feat)
        zls_feat=self.spec_proj(zls_feat)
        return zis_feat, zls_feat

