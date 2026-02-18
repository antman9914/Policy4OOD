import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch_geometric.nn import GATConv
from models.VQ import VectorQuantize
from models.SAGE import SAGEConv

class KGEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, codebook_size=500, commitment_weight=0.25):
        super(KGEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn1 = GATConv(self.input_dim, self.hidden_dim, edge_dim=self.input_dim)
        self.vq = VectorQuantize(
            dim=self.hidden_dim,
            codebook_size=codebook_size,
            decay=0.99,
            commitment_weight=commitment_weight,
            kmeans_init=True,
            threshold_ema_dead_code=2,
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, input_dim] (optional)

        Returns:
            quantized: Quantized node embeddings [num_nodes, hidden_dim]
            indices: Codebook indices for each node [num_nodes]
            vq_loss: Vector quantization loss (commitment loss)
            gat_output: Original GAT output before quantization [num_nodes, hidden_dim]
        """
        # Encode with two-layer GAT
        gat_output = self.gnn1(x, edge_index, edge_attr)
        gat_output = F.relu(gat_output)

        quantized, indices, vq_loss, _ = self.vq(gat_output)

        return quantized, indices, vq_loss, gat_output

    def get_codebook(self):
        """Return the learned codebook embeddings."""
        return self.vq.codebook

    def decode_indices(self, indices):
        """Convert codebook indices back to embeddings."""
        return self.vq.get_codes_from_indices(indices)



class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = False):
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        # Tensor, shape (batch_size, seq_len, 1)
        timestamps = timestamps.unsqueeze(dim=-1)

        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output


class Policy4OOD(nn.Module):

    def __init__(self, input_dim, args):
        super(Policy4OOD, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.output_step
        self.policy_dim = args.policy_dim
        self.num_heads = args.num_heads
        self.num_enc_layers = args.enc_layers
        self.dropout = args.dropout
        
        self.lin = nn.Linear(input_dim, self.hidden_dim - self.policy_dim)

        self.act = nn.ReLU()
        self.gcn_1 = SAGEConv(input_dim, input_dim)
        self.feat_lin = nn.Linear(input_dim, self.policy_dim)
        self.pos_emb = TimeEncoder(args.time_feat_dim)

        hidden_dim = self.hidden_dim + args.time_feat_dim
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim*4,
                dropout=self.dropout,
                norm_first=False,
                batch_first=True
            )
        self.encoder = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_enc_layers)])
        self.encoder_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(self.num_enc_layers)])

        self.kg_encoder = KGEncoder(args.text_dim, self.policy_dim)
        self.kg_lin = nn.Linear(self.policy_dim, self.policy_dim)
        self.state_lin = nn.Linear(self.policy_dim, self.policy_dim)
        self.future_lin = nn.Linear(self.policy_dim, self.policy_dim)

    def transformer_encode(self, x):
        for layer, norm in zip(self.encoder, self.encoder_norm):
            last_x = x
            x = layer(norm(x))
            x = last_x + x
        return x

    def forward(self, ts, graph, kg_set, kg_index, future_kg_index):

        input_feat = ts.squeeze()
        # Spatial GNN encoder
        x_gnn = []
        for i in range(input_feat.size(1)):
            gx_1 = self.gcn_1(input_feat[:, i, :], graph)
            x_gnn.append(gx_1)
        x_gnn_orig = torch.stack(x_gnn, dim=1)
        x_gnn = self.lin(x_gnn_orig)
        x_gnn = self.act(x_gnn)

        # Policy KG based VQ learning
        vq_loss_total = 0.
        total_num = 0
        unique_index = [np.unique(kg_index[i, :]) for i in range(kg_index.shape[0])]
        kg_mean = torch.zeros((input_feat.size(0), input_feat.size(1), self.policy_dim))
        for i in range(input_feat.size(0)):
            for index in unique_index[i]:
                # index == -1 means that there is no KG in corresponding timestamp.
                if index == -1:
                    continue
                adj, efeat, nfeat = kg_set[i][index]
                _, _, vq_loss, gat_output = self.kg_encoder(nfeat, adj, efeat)
                vq_loss_total += vq_loss
                total_num += 1
                kg_mean[i, kg_index[i] == index, :] = gat_output.mean(0, keepdim=True)
        future_unique_index = [np.unique(future_kg_index[i, :]) for i in range(future_kg_index.shape[0])]
        future_kg_mean = torch.zeros((input_feat.size(0), future_kg_index.size(1), self.policy_dim))
        for i in range(input_feat.size(0)):
            for index in future_unique_index[i]:
                if index == -1:
                    continue
                adj, efeat, nfeat = kg_set[i][index]
                _, _, vq_loss, gat_output = self.kg_encoder(nfeat, adj, efeat)
                vq_loss_total += vq_loss
                total_num += 1
                future_kg_mean[i, future_kg_index[i] == index, :] = gat_output.mean(0, keepdim=True)

        codebook = self.kg_encoder.get_codebook()
        
        # VQ code aggregation
        # Compute cosine similarity between kg_mean and codebook
        kg_mean_norm = F.normalize(kg_mean, p=2, dim=-1)
        future_kg_mean_norm = F.normalize(future_kg_mean, p=2, dim=-1)
        gnn_norm = F.normalize(self.feat_lin(x_gnn_orig), p=2, dim=-1)
        codebook_norm = F.normalize(codebook, p=2, dim=-1)

        # Compute similarity: (num_states, seq_len, codebook_size)
        similarity = torch.einsum('nsd,cd->nsc', kg_mean_norm, codebook_norm)
        future_sim = torch.einsum('nsd,cd->nsc', future_kg_mean_norm, codebook_norm)
        state_sim = torch.einsum('nsd,cd->nsc', gnn_norm, codebook_norm)

        # Apply softmax to get aggregation weights
        weights = F.softmax(similarity, dim=-1)
        state_weights = F.softmax(state_sim, dim=-1)
        weights = torch.where(kg_index.unsqueeze(-1) == -1, torch.zeros_like(weights), weights)
        state_weights = torch.where(kg_index.unsqueeze(-1) == -1, torch.zeros_like(state_weights), state_weights)

        future_weights = F.softmax(future_sim, dim=-1)
        future_weights = torch.where(future_kg_index.unsqueeze(-1) == -1, torch.zeros_like(future_weights), future_weights)

        # Aggregate codebook embeddings
        kg_aggregated = torch.einsum('nsc,cd->nsd', weights, codebook)
        state_aggr = torch.einsum('nsc,cd->nsd', state_weights, codebook)
        future_aggr = torch.einsum('nsc,cd->nsd', future_weights, codebook)

        # Project and combine kg_aggregated with x_gnn
        kg_feat = self.kg_lin(kg_aggregated)
        state_feat = self.state_lin(state_aggr)
        future_feat = self.future_lin(future_aggr)
        kg_feat = torch.cat([kg_feat[:, :-1, :], future_feat[:, -1:, :]], dim=1)
        policy_feat = kg_feat + state_feat
        x_combined = torch.cat([x_gnn, policy_feat], dim=-1)

        ts = torch.arange(input_feat.size(1)).unsqueeze(0).repeat(input_feat.size(0), 1).float()
        x_combined = torch.cat([x_combined, self.pos_emb(ts)], dim=-1)
        x_2 = self.transformer_encode(x_combined)
        pred = x_2.mean(1)

        return pred, vq_loss / total_num