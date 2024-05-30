from device import device
from bidirectional import BertModel
from torch_geometric.nn import RGCNConv
from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
PLM = 'bert-base-uncased'


class RelationalGCN(torch.nn.Module):
    def __init__(self, args, input_dim=768, hidden_channels=128,
                 layers=5):
        super(RelationalGCN, self).__init__()
        self.args = args
        num_relations = 12 if self.args.unidirectional or self.args.bidirectional else 10
        self.convs = torch.nn.ModuleList(
            [RGCNConv(in_channels=input_dim, out_channels=hidden_channels, num_relations=num_relations)] +
            [RGCNConv(in_channels=hidden_channels, out_channels=hidden_channels, num_relations=num_relations)
                for _ in range(layers-2)] +
            [RGCNConv(in_channels=hidden_channels, out_channels=hidden_channels, num_relations=num_relations)]
        )

    def forward(self, x, edge_index, edge_type, batch, inds, special_node_idx=None):
        for layer in self.convs[:-1]:
            x = F.relu(layer(x, edge_index, edge_type.view(-1)))
        x = self.convs[-1](x, edge_index, edge_type.view(-1))
        if self.args.unidirectional:
            special_mask = special_node_idx.view(-1, 1)
            z = x[special_mask[:, 0]]
            inds = torch.stack(inds, dim=-1).tolist()
            for i in range(1, len(inds)):
                v_c = 0
                for j in range(i, 0, -1):
                    v_c += batch.tolist().count(j-1)
                inds[i][0] += v_c
                inds[i][1] += v_c
            mask = torch.tensor(inds)
            head = x[mask[:, 0]]
            tail = x[mask[:, 1]]
            avg_n = torch.cat([z, head, tail], dim=-1).view(special_mask.shape[0], 3, -1)
            pool = torch.mean(avg_n, dim=1)
            return pool, z
        else:
            inds = torch.stack(inds, dim=-1).tolist()
            for i in range(1, len(inds)):
                v_c = 0
                for j in range(i, 0, -1):
                    v_c += batch.tolist().count(j-1)
                inds[i][0] += v_c
                inds[i][1] += v_c
            mask = torch.tensor(inds)
            head = x[mask[:, 0]]
            tail = x[mask[:, 1]]
            x = torch.cat([head, tail], dim=-1)
            return x


class TextRGCN(torch.nn.Module):
    def __init__(self, args, graph_input_dim=768):
        super(TextRGCN, self).__init__()
        self.hidden_channels = args.gnn_hidden
        self.GNN_only = args.GNN_only
        self.LM_only = args.LM_only
        self.gcn = RelationalGCN(args, graph_input_dim, self.hidden_channels, layers=args.gnn_layers).to(device)
        self.dropout_t = nn.Dropout(args.dropout)
        self.dropout_g = nn.Dropout(args.dropout)
        self.unidirectional = args.unidirectional
        self.bidirectional = args.bidirectional
        if args.bidirectional:
            self.text_encoder = BertModel.from_pretrained(PLM, return_dict=True, output_hidden_states=True).to(device)
        else:
            self.text_encoder = AutoModel.from_pretrained(PLM, return_dict=True, output_hidden_states=True).to(device)

        gnn_output_dim = self.hidden_channels if self.unidirectional else self.hidden_channels * 2

        if self.GNN_only:
            self.output = nn.Linear(gnn_output_dim, gnn_output_dim)
            self.output2 = nn.Linear(gnn_output_dim, 1)
        elif self.LM_only:
            self.output = nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size,)
            self.output2 = nn.Linear(self.text_encoder.config.hidden_size, 1)
        elif self.unidirectional:
            self.output = nn.Linear(self.text_encoder.config.hidden_size + gnn_output_dim * 2, self.text_encoder.config.hidden_size,)
            self.output2 = nn.Linear(self.text_encoder.config.hidden_size, 1)
        else:
            self.output = nn.Linear(self.text_encoder.config.hidden_size + gnn_output_dim, self.text_encoder.config.hidden_size + gnn_output_dim)
            self.output2 = nn.Linear(self.text_encoder.config.hidden_size + gnn_output_dim, 1)

    def forward(self, input_ids, attention_maskask, data, inds, special_node_indicies, special_token_indicies):
        batch = data.batch.to(device)
        if self.GNN_only:
            enc = self.gcn(data.x, data.edge_index, data.edge_type, batch, inds)
            enc = self.dropout_g(enc)
        elif self.LM_only:
            enc = self.text_encoder(input_ids, attention_maskask)[1]
            enc = self.dropout_t(enc)
        else:
            if self.bidirectional:
                pooled, special_node_embedding = self.gcn(data.x, data.edge_index, data.edge_type, batch, inds, special_node_idx=special_node_indicies)
                z_text = self.text_encoder(input_ids, attention_maskask, special_node_embedding=special_node_embedding, spt=special_token_indicies)[1]
                z_text = self.dropout_t(z_text)
                enc = torch.cat([z_text, special_node_embedding, pooled], dim=-1)
            elif self.unidirectional:
                pooled, special_node_embedding = self.gcn(data.x, data.edge_index, data.edge_type, batch, inds, special_node_idx=special_node_indicies)
                z_text = self.text_encoder(input_ids, attention_maskask)[1]
                z_text = self.dropout_t(z_text)
                enc = torch.cat([z_text, special_node_embedding, pooled], dim=-1)
            else:  # The else statement is entered for the Disjoint and Grounded models
                z_graph = self.gcn(data.x, data.edge_index, data.edge_type, batch, inds, special_node_idx=None)
                z_text = self.text_encoder(input_ids, attention_maskask)[1]
                z_text = self.dropout_t(z_text)
                enc = torch.cat([z_graph, z_text], dim=-1)
        out = F.relu(self.output(enc))
        out = self.output2(out)
        return out
