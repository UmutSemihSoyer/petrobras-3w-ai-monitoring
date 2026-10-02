"""
Petrobras 3W AI Monitoring - Graph Neural Network (GNN) Fault Propagation Simulator

This module models offshore well piping topology as a Graph structure (Nodes = Wells, Edges = Manifold Risers)
to predict downstream pressure transient propagation across neighboring wells.
"""

import numpy as np
import torch
import torch.nn as nn

class WellGraphConvLayer(nn.Module):
    """Graph Convolution Layer for modeling well-manifold spatial dependencies."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: Node feature matrix (num_nodes, in_features)
        adj: Normalized Adjacency Matrix (num_nodes, num_nodes)
        """
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        return torch.relu(output)


class GNNFaultPropagationModel(nn.Module):
    """
    2-Layer Graph Neural Network for offshore manifold network fault propagation.
    """
    def __init__(self, num_nodes: int = 10, in_features: int = 8, hidden_dim: int = 16, out_dim: int = 1):
        super().__init__()
        self.num_nodes = num_nodes
        self.gconv1 = WellGraphConvLayer(in_features, hidden_dim)
        self.gconv2 = WellGraphConvLayer(hidden_dim, out_dim)
        
        # Default Manifold Topo-Adjacency Matrix (10x10)
        adj = np.eye(num_nodes)
        for i in range(num_nodes - 1):
            adj[i, i + 1] = 0.5
            adj[i + 1, i] = 0.5
        
        # Normalize adjacency
        deg = np.diag(np.sum(adj, axis=1) ** -0.5)
        adj_norm = deg @ adj @ deg
        self.register_buffer("adj", torch.tensor(adj_norm, dtype=torch.float32))

    def forward(self, well_features: torch.Tensor) -> torch.Tensor:
        """
        well_features: (10, in_features) - Current telemetry state per well
        Returns: (10, 1) - Predicted fault propagation risk per well (0.0 to 1.0)
        """
        h = self.gconv1(well_features, self.adj)
        risk = torch.sigmoid(self.gconv2(h, self.adj))
        return risk

def predict_fault_propagation(well_telemetry_matrix: np.ndarray) -> np.ndarray:
    """
    Predicts fault propagation risk scores across all 10 wells in the platform manifold.
    """
    model = GNNFaultPropagationModel()
    model.eval()
    with torch.no_grad():
        feats = torch.tensor(well_telemetry_matrix, dtype=torch.float32)
        risks = model(feats).cpu().numpy().flatten()
    return risks
