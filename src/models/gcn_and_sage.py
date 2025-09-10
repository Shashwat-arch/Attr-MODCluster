import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_clusters, dropout=0.5, leaky_relu_negative_slope=0.2):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, n_clusters)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        C = F.softmax(x, dim=1)
        return C

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_clusters, dropout=0.5, leaky_relu_negative_slope=0.2):
        super(GraphSAGEModel, self).__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, n_clusters)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_negative_slope)

    def forward(self, x, edge_index):
        x = self.sage1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.sage2(x, edge_index)
        C = F.softmax(x, dim=1)
        return C

# Example usage function to select model type
def get_model(model_type, in_dim, hidden_dim, n_clusters, dropout=0.5, leaky_relu_negative_slope=0.2):
    if model_type == 'gcn':
        return GCNModel(in_dim, hidden_dim, n_clusters, dropout, leaky_relu_negative_slope)
    elif model_type == 'graphsage':
        return GraphSAGEModel(in_dim, hidden_dim, n_clusters, dropout, leaky_relu_negative_slope)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
