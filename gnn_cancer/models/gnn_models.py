import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, global_max_pool, HeteroConv, Linear
from torch_geometric.nn import LayerNorm, BatchNorm
from typing import List, Optional, Tuple, Union
import math
import logging

logger = logging.getLogger(__name__)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism with improved numerical stability."""
    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        
        self.q_proj = nn.Linear(in_channels, in_channels)
        self.k_proj = nn.Linear(in_channels, in_channels)
        self.v_proj = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, num_nodes, head_dim]
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, num_nodes, head_dim]
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim).permute(1, 0, 2)  # [num_heads, num_nodes, head_dim]
        
        # Compute attention scores with scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [num_heads, num_nodes, num_nodes]
        
        # Create adjacency matrix from edge_index
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=scores.device, dtype=torch.bool)
        adj_matrix[edge_index[0], edge_index[1]] = True
        
        # Add self-loops
        adj_matrix.fill_diagonal_(True)
        
        # Create attention mask (True for connected nodes, False for disconnected)
        mask = adj_matrix.unsqueeze(0).expand(self.num_heads, num_nodes, num_nodes)  # [num_heads, num_nodes, num_nodes]
        
        # Apply mask: set disconnected nodes to -inf
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax with numerical stability
        attn = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attn = self.dropout(attn)
        
        # Compute output
        out = torch.matmul(attn, v)  # [num_heads, num_nodes, head_dim]
        out = out.permute(1, 0, 2).contiguous().view(num_nodes, self.num_heads * self.head_dim)  # [num_nodes, num_heads * head_dim]
        out = self.out_proj(out)
        
        return out
    
    def reset_parameters(self):
        """Reset all parameters of the model."""
        self._init_weights()

class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv(x, edge_index)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = out + identity
        return out
    
    def reset_parameters(self):
        """Reset all parameters of the model."""
        self.conv.reset_parameters()
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()

class GCN(nn.Module):
    """Enhanced Graph Convolutional Network with attention, residual connections, and edge type support."""
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 num_classes: int = 2,
                 num_edge_types: int = 3):
        super(GCN, self).__init__()
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList() if use_attention else None
        self.norms = nn.ModuleList()
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels)
        # Initial layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if use_attention:
            self.attention.append(MultiHeadAttention(hidden_channels, num_heads, dropout))
        self.norms.append(LayerNorm(hidden_channels) if use_batch_norm else nn.Identity())
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            if use_residual:
                self.convs.append(ResidualBlock(hidden_channels, hidden_channels, dropout))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if use_attention:
                self.attention.append(MultiHeadAttention(hidden_channels, num_heads, dropout))
            self.norms.append(LayerNorm(hidden_channels) if use_batch_norm else nn.Identity())
        # Output layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Initial layer
        if edge_attr is not None:
            # Extract edge type from first column of edge_attr and convert to long for embedding
            edge_types = edge_attr[:, 0].long()
            edge_emb = self.edge_type_emb(edge_types)
            # Properly aggregate edge embeddings to nodes
            agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
            agg = agg.index_add(0, edge_index[1], edge_emb)
            x = self.convs[0](x, edge_index) + agg
        else:
            x = self.convs[0](x, edge_index)
        
        if self.attention is not None:
            x = self.attention[0](x, edge_index)
        x = self.norms[0](x)
        x = F.relu(x)
        
        # Hidden layers
        for i in range(1, len(self.convs)):
            if edge_attr is not None:
                # Extract edge type from first column of edge_attr and convert to long for embedding
                edge_types = edge_attr[:, 0].long()
                edge_emb = self.edge_type_emb(edge_types)
                # Properly aggregate edge embeddings to nodes
                agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
                agg = agg.index_add(0, edge_index[1], edge_emb)
                x = self.convs[i](x, edge_index) + agg
            else:
                x = self.convs[i](x, edge_index)
            
            if self.attention is not None:
                x = self.attention[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
        
        # Output layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return raw logits
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.attention is not None:
            for attn in self.attention:
                attn.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.edge_type_emb.reset_parameters()

class GraphSAGE(nn.Module):
    """Enhanced GraphSAGE with attention, residual connections, and edge type support."""
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 num_classes: int = 2,
                 num_edge_types: int = 3):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList() if use_attention else None
        self.norms = nn.ModuleList()
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels)
        # Initial layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        if use_attention:
            self.attention.append(MultiHeadAttention(hidden_channels, num_heads, dropout))
        self.norms.append(LayerNorm(hidden_channels) if use_batch_norm else nn.Identity())
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            if use_residual:
                self.convs.append(ResidualBlock(hidden_channels, hidden_channels, dropout))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if use_attention:
                self.attention.append(MultiHeadAttention(hidden_channels, num_heads, dropout))
            self.norms.append(LayerNorm(hidden_channels) if use_batch_norm else nn.Identity())
        # Output layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Initial layer
        if edge_attr is not None:
            # Extract edge type from first column of edge_attr and convert to long for embedding
            edge_types = edge_attr[:, 0].long()
            edge_emb = self.edge_type_emb(edge_types)
            # Properly aggregate edge embeddings to nodes
            agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
            agg = agg.index_add(0, edge_index[1], edge_emb)
            x = self.convs[0](x, edge_index) + agg
        else:
            x = self.convs[0](x, edge_index)
        if self.attention is not None:
            x = self.attention[0](x, edge_index)
        x = self.norms[0](x)
        x = F.relu(x)
        # Hidden layers
        for i in range(1, len(self.convs)):
            if edge_attr is not None:
                # Extract edge type from first column of edge_attr and convert to long for embedding
                edge_types = edge_attr[:, 0].long()
                edge_emb = self.edge_type_emb(edge_types)
                # Properly aggregate edge embeddings to nodes
                agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
                agg = agg.index_add(0, edge_index[1], edge_emb)
                if isinstance(self.convs[i], ResidualBlock):
                    x = self.convs[i](x, edge_index) + agg
                else:
                    x = self.convs[i](x, edge_index) + agg
            else:
                x = self.convs[i](x, edge_index)
            if self.attention is not None:
                x = self.attention[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
        # Output layers (no global mean pooling)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return raw logits
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.attention is not None:
            for attn in self.attention:
                attn.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.edge_type_emb.reset_parameters()

class GAT(nn.Module):
    """Enhanced Graph Attention Network with multi-head attention, residual connections, and edge type support."""
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_attention: bool = True,
                 use_residual: bool = True,
                 use_batch_norm: bool = True,
                 num_classes: int = 2,
                 num_edge_types: int = 3):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.attention = nn.ModuleList() if use_attention else None
        self.norms = nn.ModuleList()
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_channels * num_heads)
        # Initial layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads))
        if use_attention:
            self.attention.append(MultiHeadAttention(hidden_channels * num_heads, num_heads, dropout))
        self.norms.append(LayerNorm(hidden_channels * num_heads) if use_batch_norm else nn.Identity())
        # Hidden layers with residual connections
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads))
            if use_attention:
                self.attention.append(MultiHeadAttention(hidden_channels * num_heads, num_heads, dropout))
            self.norms.append(LayerNorm(hidden_channels * num_heads) if use_batch_norm else nn.Identity())
        # Output layers
        self.fc1 = nn.Linear(hidden_channels * num_heads, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        # Initial layer
        if edge_attr is not None:
            # Extract edge type from first column of edge_attr and convert to long for embedding
            edge_types = edge_attr[:, 0].long()
            edge_emb = self.edge_type_emb(edge_types)
            # Properly aggregate edge embeddings to nodes
            agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
            agg = agg.index_add(0, edge_index[1], edge_emb)
            x = self.convs[0](x, edge_index) + agg
        else:
            x = self.convs[0](x, edge_index)
        if self.attention is not None:
            x = self.attention[0](x, edge_index)
        x = self.norms[0](x)
        x = F.relu(x)
        # Hidden layers
        for i in range(1, len(self.convs)):
            if edge_attr is not None:
                # Extract edge type from first column of edge_attr and convert to long for embedding
                edge_types = edge_attr[:, 0].long()
                edge_emb = self.edge_type_emb(edge_types)
                # Properly aggregate edge embeddings to nodes
                agg = torch.zeros(x.size(0), edge_emb.size(1), device=x.device)
                agg = agg.index_add(0, edge_index[1], edge_emb)
                x = self.convs[i](x, edge_index) + agg
            else:
                x = self.convs[i](x, edge_index)
            if self.attention is not None:
                x = self.attention[i](x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
        # Output layers (no global mean pooling)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return raw logits
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.attention is not None:
            for attn in self.attention:
                attn.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.edge_type_emb.reset_parameters()

class EnhancedHeteroGNN(torch.nn.Module):
    """Enhanced heterogeneous GNN with PPI and pathway data support."""
    
    def __init__(self, in_channels_dict, hidden_channels, out_channels, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Define the heterogeneous convolution layers with PPI support
        self.conv1 = HeteroConv({
            # Gene-GO relationships
            ('gene', 'associated_with', 'go'): GATConv(
                (in_channels_dict['gene'], in_channels_dict['go']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (in_channels_dict['go'], in_channels_dict['gene']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            # Gene-PubMed relationships
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (in_channels_dict['gene'], in_channels_dict['pubmed']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (in_channels_dict['pubmed'], in_channels_dict['gene']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            # Gene-Gene PPI relationships (if available)
            ('gene', 'interacts_with', 'gene'): GATConv(
                in_channels_dict['gene'], 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        self.conv2 = HeteroConv({
            # Gene-GO relationships
            ('gene', 'associated_with', 'go'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            # Gene-PubMed relationships
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            # Gene-Gene PPI relationships
            ('gene', 'interacts_with', 'gene'): GATConv(
                hidden_channels * num_heads, 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        # Add projection layers for each node type
        self.proj = torch.nn.ModuleDict({
            'gene': torch.nn.Linear(in_channels_dict['gene'], hidden_channels * num_heads),
            'go': torch.nn.Linear(in_channels_dict['go'], hidden_channels * num_heads),
            'pubmed': torch.nn.Linear(in_channels_dict['pubmed'], hidden_channels * num_heads),
        })
        
        # Attention mechanism for combining different types of gene representations
        self.gene_attention = torch.nn.MultiheadAttention(
            embed_dim=hidden_channels * num_heads, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer for gene classification
        self.lin = Linear(hidden_channels * num_heads, out_channels)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # First layer
        prev_x_dict = x_dict.copy()
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        # For any missing node type, project previous features to correct size
        for key in prev_x_dict:
            if x_dict.get(key) is None:
                x_dict[key] = self.proj[key](prev_x_dict[key])
        
        # Apply batch normalization and activation
        x_dict = {key: F.leaky_relu(self.bn1(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Second layer
        prev_x_dict2 = x_dict.copy()
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x for key, x in x_dict.items() if x is not None}
        
        # Apply batch normalization and activation
        x_dict = {key: F.leaky_relu(self.bn2(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Ensure 'gene' is present for output
        gene_features = x_dict['gene'] if 'gene' in x_dict else prev_x_dict2['gene']
        
        # Apply attention mechanism to gene features if we have multiple gene representations
        if gene_features.dim() == 2:
            # Reshape for attention: [num_genes, hidden_dim] -> [1, num_genes, hidden_dim]
            gene_features_attn = gene_features.unsqueeze(0)
            gene_features_attn, _ = self.gene_attention(
                gene_features_attn, gene_features_attn, gene_features_attn
            )
            gene_features = gene_features_attn.squeeze(0)
        
        # Return only gene node predictions
        return self.lin(gene_features)

class MultiModalGNN(torch.nn.Module):
    """Multi-modal GNN that combines different types of biological data."""
    
    def __init__(self, in_channels_dict, hidden_channels, out_channels, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Separate encoders for different data modalities
        self.gene_encoder = torch.nn.Linear(in_channels_dict['gene'], hidden_channels)
        self.go_encoder = torch.nn.Linear(in_channels_dict['go'], hidden_channels)
        self.pubmed_encoder = torch.nn.Linear(in_channels_dict['pubmed'], hidden_channels)
        
        # Graph convolution layers
        self.conv1 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv(
                (hidden_channels, hidden_channels), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (hidden_channels, hidden_channels), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (hidden_channels, hidden_channels), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (hidden_channels, hidden_channels), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'interacts_with', 'gene'): GATConv(
                hidden_channels, 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        self.conv2 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'interacts_with', 'gene'): GATConv(
                hidden_channels * num_heads, 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        # Multi-modal fusion layer
        self.fusion_layer = torch.nn.Linear(hidden_channels * num_heads * 3, hidden_channels * num_heads)
        
        # Output layer
        self.lin = Linear(hidden_channels * num_heads, out_channels)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # Encode different modalities
        encoded_x_dict = {
            'gene': F.relu(self.gene_encoder(x_dict['gene'])),
            'go': F.relu(self.go_encoder(x_dict['go'])),
            'pubmed': F.relu(self.pubmed_encoder(x_dict['pubmed']))
        }
        
        # First graph convolution layer
        x_dict = self.conv1(encoded_x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(self.bn1(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Second graph convolution layer
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(self.bn2(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Multi-modal fusion for gene nodes
        gene_features = x_dict['gene']
        
        # Combine with GO and PubMed context
        go_context = torch.mean(x_dict['go'], dim=0) if 'go' in x_dict else torch.zeros_like(gene_features)
        pubmed_context = torch.mean(x_dict['pubmed'], dim=0) if 'pubmed' in x_dict else torch.zeros_like(gene_features)
        
        # Concatenate and fuse
        fused_features = torch.cat([gene_features, go_context, pubmed_context], dim=-1)
        fused_features = F.relu(self.fusion_layer(fused_features))
        
        return self.lin(fused_features)

class AttentionHeteroGNN(torch.nn.Module):
    """Heterogeneous GNN with attention mechanisms for better interpretability."""
    
    def __init__(self, in_channels_dict, hidden_channels, out_channels, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Attention weights for different edge types
        self.edge_attention = torch.nn.Parameter(torch.ones(3))  # GO, PubMed, PPI
        
        # Graph convolution layers
        self.conv1 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv(
                (in_channels_dict['gene'], in_channels_dict['go']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (in_channels_dict['go'], in_channels_dict['gene']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (in_channels_dict['gene'], in_channels_dict['pubmed']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (in_channels_dict['pubmed'], in_channels_dict['gene']), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'interacts_with', 'gene'): GATConv(
                in_channels_dict['gene'], 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        self.conv2 = HeteroConv({
            ('gene', 'associated_with', 'go'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('go', 'rev_associated_with', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'cited_in', 'pubmed'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('pubmed', 'rev_cited_in', 'gene'): GATConv(
                (hidden_channels * num_heads, hidden_channels * num_heads), 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
            ('gene', 'interacts_with', 'gene'): GATConv(
                hidden_channels * num_heads, 
                hidden_channels, heads=num_heads, add_self_loops=False, dropout=dropout
            ),
        })
        
        # Projection layers
        self.proj = torch.nn.ModuleDict({
            'gene': torch.nn.Linear(in_channels_dict['gene'], hidden_channels * num_heads),
            'go': torch.nn.Linear(in_channels_dict['go'], hidden_channels * num_heads),
            'pubmed': torch.nn.Linear(in_channels_dict['pubmed'], hidden_channels * num_heads),
        })
        
        # Output layer
        self.lin = Linear(hidden_channels * num_heads, out_channels)
        
        # Batch normalization
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels * num_heads)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        # First layer
        prev_x_dict = x_dict.copy()
        x_dict = self.conv1(x_dict, edge_index_dict)
        
        # For any missing node type, project previous features to correct size
        for key in prev_x_dict:
            if x_dict.get(key) is None:
                x_dict[key] = self.proj[key](prev_x_dict[key])
        
        # Apply batch normalization and activation
        x_dict = {key: F.leaky_relu(self.bn1(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Second layer
        prev_x_dict2 = x_dict.copy()
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x for key, x in x_dict.items() if x is not None}
        
        # Apply batch normalization and activation
        x_dict = {key: F.leaky_relu(self.bn2(x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}
        
        # Ensure 'gene' is present for output
        gene_features = x_dict['gene'] if 'gene' in x_dict else prev_x_dict2['gene']
        
        # Return only gene node predictions
        return self.lin(gene_features)
    
    def get_edge_attention_weights(self):
        """Return attention weights for different edge types."""
        return torch.softmax(self.edge_attention, dim=0)

def get_model(model_name: str, in_channels: int, out_channels: int = 2, **kwargs) -> nn.Module:
    """Get a GNN model by name with specified configuration.
    
    Args:
        model_name: Name of the model ('GCN', 'GraphSAGE', or 'GAT')
        in_channels: Number of input features
        out_channels: Number of output classes (default: 2)
        **kwargs: Additional model configuration parameters:
            - hidden_channels: Number of hidden channels (default: 64)
            - num_layers: Number of GNN layers (default: 2)
            - num_heads: Number of attention heads for GAT (default: 4)
            - dropout: Dropout rate (default: 0.2)
            - use_attention: Whether to use attention mechanism (default: True)
            - use_residual: Whether to use residual connections (default: True)
            - use_batch_norm: Whether to use batch normalization (default: True)
    
    Returns:
        An instance of the specified GNN model
    """
    # Default configuration
    config = {
        'hidden_channels': 64,
        'num_layers': 2,
        'num_heads': 4,
        'dropout': 0.2,
        'use_attention': True,
        'use_residual': True,
        'use_batch_norm': True
    }
    
    # Update configuration with provided kwargs
    config.update(kwargs)
    
    if model_name == 'GCN':
        return GCN(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
            use_residual=config['use_residual'],
            use_batch_norm=config['use_batch_norm'],
            num_classes=out_channels
        )
    elif model_name == 'GraphSAGE':
        return GraphSAGE(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
            use_residual=config['use_residual'],
            use_batch_norm=config['use_batch_norm'],
            num_classes=out_channels
        )
    elif model_name == 'GAT':
        return GAT(
            in_channels=in_channels,
            hidden_channels=config['hidden_channels'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            use_attention=config['use_attention'],
            use_residual=config['use_residual'],
            use_batch_norm=config['use_batch_norm'],
            num_classes=out_channels
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}") 