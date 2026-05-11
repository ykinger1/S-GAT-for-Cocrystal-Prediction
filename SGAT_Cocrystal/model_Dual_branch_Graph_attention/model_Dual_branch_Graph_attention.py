import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch


class GATEncoder(nn.Module):
    """
    Molecular graph encoder based on Graph Attention Network (GAT) for extracting node-level features.

    Args:
        in_channels (int): Dimension of atomic input features (default: 39)
        hidden_channels (int): Dimension of GAT hidden layers (default: 256)
        out_channels (int): Dimension of GAT output features (node-level) (default: 128)
        heads (int): Number of multi-head attention heads (default: 4)
        dropout (float): Dropout ratio (default: 0.1)
        concat (bool): Whether to concatenate multi-head outputs in the first layer (default: True)
    """

    def __init__(
            self,
            in_channels: int = 39,
            hidden_channels: int = 256,
            out_channels: int = 128,
            heads: int = 4,
            dropout: float = 0.1,
            concat: bool = True
    ):
        super().__init__()
        # First GAT layer: multi-head attention with concatenated outputs
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        # Second GAT layer: single-head attention for final node features
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads if concat else hidden_channels,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GAT encoder.

        Args:
            x (torch.Tensor): Atomic feature matrix with shape [num_atoms, in_channels]
            edge_index (torch.Tensor): Edge index matrix with shape [2, 2*num_bonds]
                (bidirectional storage for undirected graphs)

        Returns:
            torch.Tensor: Updated node features with shape [num_atoms, out_channels]
        """
        # First GAT layer + activation + Dropout
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # ELU activation as used in the original GAT paper
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GAT layer (no activation, processed after readout)
        x = self.conv2(x, edge_index)
        return x


class DualBranchGraphAttentionModel(nn.Module):
    """
    Co-crystal prediction model based on dual-branch Graph Attention Network (GAT).

    The model uses a shared GAT encoder to extract graph-level features for two molecules,
    concatenates the features, and feeds them into an MLP classifier for co-crystal prediction.

    Args:
        atom_in_channels (int): Dimension of atomic input features (default: 39)
        gat_hidden_channels (int): Dimension of GAT hidden layers (default: 256)
        gat_out_channels (int): Dimension of GAT output features (node-level) (default: 128)
        gat_heads (int): Number of multi-head attention heads in GAT (default: 4)
        gat_dropout (float): Dropout ratio for GAT layers (default: 0.1)
        readout (str): Graph readout pooling method, must be "mean" (average pooling)
            or "sum" (sum pooling) (default: "sum")
        mlp_hidden_channels (int): Dimension of MLP hidden layers (default: 1024)
        mlp_dropout (float): Dropout ratio for MLP layers (default: 0.3)
    """

    def __init__(
            self,
            # GAT encoder parameters
            atom_in_channels: int = 39,
            gat_hidden_channels: int = 256,
            gat_out_channels: int = 128,
            gat_heads: int = 4,
            gat_dropout: float = 0.1,
            # Readout parameters
            readout: str = "sum",
            # MLP classifier head parameters
            mlp_hidden_channels: int = 1024,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        # Shared-weight GAT encoder (both molecules share the same encoder)
        self.gat_encoder = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True
        )

        # Readout pooling layer
        self.readout = readout
        if readout not in ["mean", "sum"]:
            raise ValueError(f"Readout type must be 'mean' or 'sum', got {readout}")

        # MLP classifier head (concatenates graph representations of two molecules)
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)  # Output logits (no sigmoid applied)
        )

    def forward(
            self,
            data1: Data | Batch,  # PyG Data/Batch object for the first molecule
            data2: Data | Batch  # PyG Data/Batch object for the second molecule
    ) -> torch.Tensor:
        """
        Forward pass of the dual-branch GAT model.

        Args:
            data1 (Data | Batch): PyG Data/Batch object for the first molecule, containing:
                - x: Atomic feature matrix
                - edge_index: Edge index matrix
                - batch (optional): Batch index (required for Batch objects, omitted for single-graph Data)
            data2 (Data | Batch): PyG Data/Batch object for the second molecule (same structure as data1)

        Returns:
            torch.Tensor: Classification logits with shape [batch_size, 1].
                Use BCEWithLogitsLoss for training and sigmoid for inference.
        """
        # Extract graph representation of the first molecule
        x1, edge_index1 = data1.x, data1.edge_index
        batch1 = data1.batch if hasattr(data1, "batch") else None  # batch is None for single graph
        node_feat1 = self.gat_encoder(x1, edge_index1)
        # Readout: node features -> graph-level features
        if self.readout == "mean":
            graph_feat1 = global_mean_pool(node_feat1, batch1)  # [batch_size, gat_out_channels]
        else:
            graph_feat1 = global_add_pool(node_feat1, batch1)

        # Extract graph representation of the second molecule (shared encoder)
        x2, edge_index2 = data2.x, data2.edge_index
        batch2 = data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder(x2, edge_index2)
        if self.readout == "mean":
            graph_feat2 = global_mean_pool(node_feat2, batch2)
        else:
            graph_feat2 = global_add_pool(node_feat2, batch2)

        # Concatenate graph representations + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)  # [batch_size, 2*gat_out_channels]
        logits = self.mlp(combined_feat)
        return logits