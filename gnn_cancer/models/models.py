# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_mean_pool

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.5, num_edge_types=3):
        super(GCNModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs[:-1]):
            if edge_attr is not None:
                edge_emb = self.edge_type_emb(edge_attr)
                x = conv(x, edge_index) + torch.zeros_like(x)
                # Add edge type embedding to target nodes
                x = x + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb)
            else:
                x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = self.convs[-1](x, edge_index) + torch.zeros_like(x)
            x = x + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb)
        else:
            x = self.convs[-1](x, edge_index)
        return x
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.edge_type_emb.reset_parameters()


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.5, num_edge_types=3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, normalize=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        # Skip connections
        self.skip1 = nn.Linear(in_channels, hidden_channels)
        self.skip2 = nn.Linear(hidden_channels, hidden_channels)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        # First GraphSAGE layer with skip connection
        identity = self.skip1(x)
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.relu(self.conv1(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.relu(self.conv1(x, edge_index))
        x = x + identity  # Skip connection
        x = self.dropout(x)
        # Second GraphSAGE layer with skip connection
        identity = self.skip2(x)
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.relu(self.conv2(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.relu(self.conv2(x, edge_index))
        x = x + identity  # Skip connection
        x = self.dropout(x)
        # Third GraphSAGE layer
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.relu(self.conv3(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        # Classification layer
        x = self.classifier(x)
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.classifier.reset_parameters()
        self.skip1.reset_parameters()
        self.skip2.reset_parameters()
        self.edge_type_emb.reset_parameters()


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_heads=8, dropout=0.5, num_edge_types=3):
        super(GATModel, self).__init__()
        # Multi-head attention layers
        self.conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        # GAT layers with ELU activation
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.elu(self.conv1(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.elu(self.conv2(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        if edge_attr is not None:
            edge_emb = self.edge_type_emb(edge_attr)
            x = F.elu(self.conv3(x, edge_index) + torch.zeros_like(x).index_add(0, edge_index[1], edge_emb))
        else:
            x = F.elu(self.conv3(x, edge_index))
        x = self.dropout(x)
        # Classification layer
        x = self.classifier(x)
        return x
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.classifier.reset_parameters()
        self.edge_type_emb.reset_parameters()