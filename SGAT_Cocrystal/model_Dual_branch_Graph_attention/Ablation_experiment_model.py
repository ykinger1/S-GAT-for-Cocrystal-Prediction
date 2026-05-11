import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
import warnings

warnings.filterwarnings("ignore")


# ===================== Base Encoder Definitions =====================
class GATEncoder(nn.Module):
    """Original GAT encoder (used in Baseline/A4/A8 experiments)

    Args:
        in_channels (int): Number of input channels for node features
        hidden_channels (int): Number of hidden channels in first GAT layer
        out_channels (int): Number of output channels in second GAT layer
        heads (int): Number of attention heads in first GAT layer
        dropout (float): Dropout probability
        concat (bool): Whether to concatenate multi-head features (True for first layer)
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
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads if concat else hidden_channels,
            out_channels=out_channels,
            heads=1,
            dropout=dropout,
            concat=False
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of GAT encoder

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (torch.Tensor): Graph edge index with shape [2, num_edges]

        Returns:
            torch.Tensor: Node embeddings with shape [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GCNEncoder(nn.Module):
    """A1 Experiment: GCN encoder replacing GAT

    Args:
        in_channels (int): Number of input channels for node features
        hidden_channels (int): Number of hidden channels in first GCN layer
        out_channels (int): Number of output channels in second GCN layer
        dropout (float): Dropout probability
    """

    def __init__(
            self,
            in_channels: int = 39,
            hidden_channels: int = 64,
            out_channels: int = 128,
            dropout: float = 0.2
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of GCN encoder

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels]
            edge_index (torch.Tensor): Graph edge index with shape [2, num_edges]

        Returns:
            torch.Tensor: Node embeddings with shape [num_nodes, out_channels]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)  # ReLU is commonly used in GCN
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ===================== Baseline Model =====================
class DualBranchGraphAttentionModel(nn.Module):
    """Basic dual-branch GAT model (Baseline)

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        gat_hidden_channels (int): Hidden channels in GAT encoder
        gat_out_channels (int): Output channels in GAT encoder
        gat_heads (int): Number of attention heads in GAT
        gat_dropout (float): Dropout probability in GAT encoder
        readout (str): Graph readout method ('mean' or 'sum')
        mlp_hidden_channels (int): Hidden channels in MLP classifier
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            gat_hidden_channels: int = 256,
            gat_out_channels: int = 128,
            gat_heads: int = 4,
            gat_dropout: float = 0.1,
            readout: str = "sum",
            mlp_hidden_channels: int = 1024,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        self.gat_encoder = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True
        )
        self.readout = readout
        if readout not in ["mean", "sum"]:
            raise ValueError(f"Readout type must be 'mean' or 'sum', got {readout}")
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of baseline dual-branch GAT model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Process molecule 1
        x1, edge_index1 = data1.x, data1.edge_index
        batch1 = data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gat_encoder(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        # Process molecule 2
        x2, edge_index2 = data2.x, data2.edge_index
        batch2 = data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        # Concatenate features + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


# ===================== Ablation Experiment Models =====================
class BaselineModel(DualBranchGraphAttentionModel):
    """Baseline: Original model (no modifications)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class A1Model(nn.Module):
    """A1 Experiment: Replace GATConv with GCNConv

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        gcn_hidden_channels (int): Hidden channels in GCN encoder
        gcn_out_channels (int): Output channels in GCN encoder
        gcn_dropout (float): Dropout probability in GCN encoder
        readout (str): Graph readout method ('mean' or 'sum')
        mlp_hidden_channels (int): Hidden channels in MLP classifier
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            gcn_hidden_channels: int = 64,
            gcn_out_channels: int = 128,
            gcn_dropout: float = 0.2,
            readout: str = "mean",
            mlp_hidden_channels: int = 256,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        self.gcn_encoder = GCNEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gcn_hidden_channels,
            out_channels=gcn_out_channels,
            dropout=gcn_dropout
        )
        self.readout = readout
        self.mlp = nn.Sequential(
            nn.Linear(gcn_out_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A1 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Encode molecule 1
        x1, edge_index1 = data1.x, data1.edge_index
        batch1 = data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gcn_encoder(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        # Encode molecule 2
        x2, edge_index2 = data2.x, data2.edge_index
        batch2 = data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gcn_encoder(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        # Concatenate + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


class A2Model(nn.Module):
    """A2 Experiment: Remove graph convolution, directly apply mean pooling on atomic features then concatenate

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        mlp_hidden_channels (int): Hidden channels in MLP classifier
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            mlp_hidden_channels: int = 256,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(atom_in_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A2 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Molecule 1: direct mean pooling on atomic features
        x1, batch1 = data1.x, data1.batch if hasattr(data1, "batch") else None
        graph_feat1 = global_mean_pool(x1, batch1)

        # Molecule 2: direct mean pooling on atomic features
        x2, batch2 = data2.x, data2.batch if hasattr(data2, "batch") else None
        graph_feat2 = global_mean_pool(x2, batch2)

        # Concatenate + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


class A3Model(nn.Module):
    """A3 Experiment: Atomic features first pass through independent MLP, then pooling (no graph convolution)

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        atom_mlp_hidden (int): Hidden channels in atomic feature MLP
        mlp_hidden_channels (int): Hidden channels in classification MLP
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            atom_mlp_hidden: int = 128,
            mlp_hidden_channels: int = 256,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        # Independent MLP for atomic features
        self.atom_mlp = nn.Sequential(
            nn.Linear(atom_in_channels, atom_mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(atom_mlp_hidden, atom_mlp_hidden)
        )
        # Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(atom_mlp_hidden * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A3 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Molecule 1: atomic features pass through MLP then pooling
        x1, batch1 = data1.x, data1.batch if hasattr(data1, "batch") else None
        x1_mlp = self.atom_mlp(x1)
        graph_feat1 = global_mean_pool(x1_mlp, batch1)

        # Molecule 2: atomic features pass through MLP then pooling
        x2, batch2 = data2.x, data2.batch if hasattr(data2, "batch") else None
        x2_mlp = self.atom_mlp(x2)
        graph_feat2 = global_mean_pool(x2_mlp, batch2)

        # Concatenate + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


class A4Model(nn.Module):
    """A4 Experiment: Build two independent GAT encoders with non-shared parameters

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        gat_hidden_channels (int): Hidden channels in GAT encoder
        gat_out_channels (int): Output channels in GAT encoder
        gat_heads (int): Number of attention heads in GAT
        gat_dropout (float): Dropout probability in GAT encoder
        readout (str): Graph readout method ('mean' or 'sum')
        mlp_hidden_channels (int): Hidden channels in MLP classifier
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            gat_hidden_channels: int = 64,
            gat_out_channels: int = 128,
            gat_heads: int = 4,
            gat_dropout: float = 0.2,
            readout: str = "mean",
            mlp_hidden_channels: int = 256,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        # Two independent GAT encoders
        self.gat_encoder1 = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True
        )
        self.gat_encoder2 = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True
        )
        self.readout = readout
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A4 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Molecule 1: independent encoder 1
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gat_encoder1(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        # Molecule 2: independent encoder 2
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder2(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        # Concatenate + MLP classification
        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


class A5Model(nn.Module):
    """A5 Experiment: Merge x/edge_index of two molecules into one graph, encode uniformly then input to MLP

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        gat_hidden_channels (int): Hidden channels in GAT encoder
        gat_out_channels (int): Output channels in GAT encoder
        gat_heads (int): Number of attention heads in GAT
        gat_dropout (float): Dropout probability in GAT encoder
        readout (str): Graph readout method ('mean' or 'sum')
        mlp_hidden_channels (int): Hidden channels in MLP classifier
        mlp_dropout (float): Dropout probability in MLP
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            gat_hidden_channels: int = 64,
            gat_out_channels: int = 128,
            gat_heads: int = 4,
            gat_dropout: float = 0.2,
            readout: str = "mean",
            mlp_hidden_channels: int = 256,
            mlp_dropout: float = 0.3
    ):
        super().__init__()
        self.gat_encoder = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=gat_dropout,
            concat=True
        )
        self.readout = readout
        # MLP input dimension is single graph feature dimension
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def _merge_graph_batch(self, batch1: Batch, batch2: Batch) -> Data:
        """
        Core fix: Batch-wise pairwise merging of two molecular graphs
        Input: batch1 (N graphs), batch2 (N graphs)
        Output: Merged Batch (N graphs, each graph = merged two molecules at corresponding position)
        """
        # Get number of graphs in batch
        num_graphs = batch1.num_graphs
        device = batch1.x.device

        all_x = []
        all_edge_index = []
        all_batch = []

        # Merge each pair of molecules in the batch
        node_offset = 0
        for i in range(num_graphs):
            # Extract i-th graph for molecule 1 and molecule 2
            g1 = batch1.get_example(i)
            g2 = batch2.get_example(i)

            # Merge node features
            x = torch.cat([g1.x, g2.x], dim=0)
            all_x.append(x)

            # Merge edge indices (offset for molecule 2 edge indices)
            edge_index = torch.cat([g1.edge_index, g2.edge_index + g1.x.shape[0]], dim=1)
            all_edge_index.append(edge_index + node_offset)

            # Merge batch indices (current merged graph belongs to i-th sample)
            batch = torch.full((x.shape[0],), i, device=device)
            all_batch.append(batch)

            # Update node offset
            node_offset += x.shape[0]

        # Concatenate all merged data
        x = torch.cat(all_x, dim=0)
        edge_index = torch.cat(all_edge_index, dim=1)
        batch = torch.cat(all_batch, dim=0)

        return Data(x=x, edge_index=edge_index, batch=batch)

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A5 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Batch-wise pairwise merging of two molecules into one graph (core fix)
        merged_batch = self._merge_graph_batch(data1, data2)

        # Encode merged graph
        node_feat = self.gat_encoder(merged_batch.x, merged_batch.edge_index)
        # Correct pooling: output [batch_size, gat_out_channels]
        graph_feat = global_mean_pool(node_feat, merged_batch.batch)

        # MLP classification: output [batch_size, 1]
        logits = self.mlp(graph_feat)
        return logits


class A6Model(DualBranchGraphAttentionModel):
    """A6 Experiment: Feature addition instead of feature concatenation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Rewrite MLP: input dimension changed to single graph feature dimension
        gat_out_channels = kwargs.get("gat_out_channels", 128)
        mlp_hidden_channels = kwargs.get("mlp_hidden_channels", 256)
        mlp_dropout = kwargs.get("mlp_dropout", 0.3)
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A6 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Extract graph features of two molecules (same as Baseline)
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gat_encoder(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        # Feature addition instead of concatenation
        combined_feat = graph_feat1 + graph_feat2
        logits = self.mlp(combined_feat)
        return logits


class A7Model(DualBranchGraphAttentionModel):
    """A7 Experiment: Feature multiplication instead of feature concatenation"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Rewrite MLP: input dimension changed to single graph feature dimension
        gat_out_channels = kwargs.get("gat_out_channels", 128)
        mlp_hidden_channels = kwargs.get("mlp_hidden_channels", 256)
        mlp_dropout = kwargs.get("mlp_dropout", 0.3)
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A7 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Extract graph features of two molecules (same as Baseline)
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gat_encoder(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        # Feature multiplication instead of concatenation
        combined_feat = graph_feat1 * graph_feat2
        logits = self.mlp(combined_feat)
        return logits


class A8Model(nn.Module):
    """A8 Experiment: Remove all Dropout layers

    Args:
        atom_in_channels (int): Number of input channels for atomic features
        gat_hidden_channels (int): Hidden channels in GAT encoder
        gat_out_channels (int): Output channels in GAT encoder
        gat_heads (int): Number of attention heads in GAT
        readout (str): Graph readout method ('mean' or 'sum')
        mlp_hidden_channels (int): Hidden channels in MLP classifier
    """

    def __init__(
            self,
            atom_in_channels: int = 39,
            gat_hidden_channels: int = 64,
            gat_out_channels: int = 128,
            gat_heads: int = 4,
            readout: str = "mean",
            mlp_hidden_channels: int = 256
    ):
        super().__init__()
        # GAT encoder (Dropout set to 0)
        self.gat_encoder = GATEncoder(
            in_channels=atom_in_channels,
            hidden_channels=gat_hidden_channels,
            out_channels=gat_out_channels,
            heads=gat_heads,
            dropout=0.0,  # Remove Dropout
            concat=True
        )
        self.readout = readout
        # MLP classifier (remove all Dropout)
        self.mlp = nn.Sequential(
            nn.Linear(gat_out_channels * 2, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, mlp_hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels // 2, 1)
        )

    def forward(self, data1: Data | Batch, data2: Data | Batch) -> torch.Tensor:
        """Forward pass of A8 model

        Args:
            data1 (Data | Batch): First molecule graph data/batch
            data2 (Data | Batch): Second molecule graph data/batch

        Returns:
            torch.Tensor: Prediction logits with shape [batch_size, 1]
        """
        # Same as Baseline (without Dropout)
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch if hasattr(data1, "batch") else None
        node_feat1 = self.gat_encoder(x1, edge_index1)
        graph_feat1 = global_mean_pool(node_feat1, batch1) if self.readout == "mean" else global_add_pool(node_feat1, batch1)

        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch if hasattr(data2, "batch") else None
        node_feat2 = self.gat_encoder(x2, edge_index2)
        graph_feat2 = global_mean_pool(node_feat2, batch2) if self.readout == "mean" else global_add_pool(node_feat2, batch2)

        combined_feat = torch.cat([graph_feat1, graph_feat2], dim=1)
        logits = self.mlp(combined_feat)
        return logits


# ===================== Ablation Experiment Test Main Function =====================
def generate_synthetic_data(batch_size: int = 2) -> tuple[Batch, Batch]:
    """Generate synthetic PyG Batch data (for testing)

    Args:
        batch_size (int): Number of samples in batch

    Returns:
        tuple[Batch, Batch]: Two batches of synthetic molecular graph data
    """
    data_list1, data_list2 = [], []
    for _ in range(batch_size):
        # Randomly generate Data for molecule 1
        num_atoms1 = torch.randint(5, 20, (1,)).item()
        x1 = torch.randn(num_atoms1, 39)
        num_bonds1 = torch.randint(5, 30, (1,)).item()
        edge_index1 = torch.randint(0, num_atoms1, (2, num_bonds1))
        data_list1.append(Data(x=x1, edge_index=edge_index1))

        # Randomly generate Data for molecule 2
        num_atoms2 = torch.randint(5, 20, (1,)).item()
        x2 = torch.randn(num_atoms2, 39)
        num_bonds2 = torch.randint(5, 30, (1,)).item()
        edge_index2 = torch.randint(0, num_atoms2, (2, num_bonds2))
        data_list2.append(Data(x=x2, edge_index=edge_index2))

    # Pack into Batch
    batch1 = Batch.from_data_list(data_list1)
    batch2 = Batch.from_data_list(data_list2)
    return batch1, batch2


def ablation_experiment() -> dict:
    """Ablation experiment main function: Test forward propagation of all experimental models

    Returns:
        dict: Experiment results containing status, output shape and error info (if any)
    """
    # General configuration
    config = {
        "atom_in_channels": 39,
        "gat_hidden_channels": 64,
        "gat_out_channels": 128,
        "gat_heads": 4,
        "gat_dropout": 0.2,
        "readout": "mean",
        "mlp_hidden_channels": 256,
        "mlp_dropout": 0.3
    }

    # Generate test data
    batch1, batch2 = generate_synthetic_data(batch_size=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch1, batch2 = batch1.to(device), batch2.to(device)

    # Define all experimental groups
    experiments = {
        "Baseline": BaselineModel(**config),
        "A1": A1Model(
            atom_in_channels=config["atom_in_channels"],
            gcn_hidden_channels=config["gat_hidden_channels"],
            gcn_out_channels=config["gat_out_channels"],
            gcn_dropout=config["gat_dropout"],
            readout=config["readout"],
            mlp_hidden_channels=config["mlp_hidden_channels"],
            mlp_dropout=config["mlp_dropout"]
        ),
        "A2": A2Model(
            atom_in_channels=config["atom_in_channels"],
            mlp_hidden_channels=config["mlp_hidden_channels"],
            mlp_dropout=config["mlp_dropout"]
        ),
        "A3": A3Model(
            atom_in_channels=config["atom_in_channels"],
            atom_mlp_hidden=config["gat_out_channels"],
            mlp_hidden_channels=config["mlp_hidden_channels"],
            mlp_dropout=config["mlp_dropout"]
        ),
        "A4": A4Model(**config),
        "A5": A5Model(**config),
        "A6": A6Model(**config),
        "A7": A7Model(**config),
        "A8": A8Model(
            atom_in_channels=config["atom_in_channels"],
            gat_hidden_channels=config["gat_hidden_channels"],
            gat_out_channels=config["gat_out_channels"],
            gat_heads=config["gat_heads"],
            readout=config["readout"],
            mlp_hidden_channels=config["mlp_hidden_channels"]
        )
    }

    # Test each experimental group
    results = {}
    print("=" * 50)
    print("Starting ablation experiment testing...")
    print(f"Testing device: {device} | Batch size: {batch1.num_graphs}")
    print("=" * 50)

    for exp_name, model in experiments.items():
        try:
            model = model.to(device).eval()
            with torch.no_grad():
                logits = model(batch1, batch2)

            # Verify output dimension
            assert logits.shape == (batch1.num_graphs, 1), \
                f"{exp_name} output dimension error: {logits.shape}, expected ({batch1.num_graphs}, 1)"

            results[exp_name] = {
                "status": "SUCCESS",
                "output_shape": logits.shape,
                "sample_output": logits.cpu().numpy().tolist()
            }
            print(f"✅ {exp_name}: Test passed | Output shape: {logits.shape}")

        except Exception as e:
            results[exp_name] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ {exp_name}: Test failed | Error: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("Ablation experiment test summary:")
    for exp_name, res in results.items():
        if res["status"] == "SUCCESS":
            print(f"{exp_name:<8}: Success | Output shape: {res['output_shape']}")
        else:
            print(f"{exp_name:<8}: Failed | Error: {res['error'][:100]}...")
    print("=" * 50)

    return results


if __name__ == "__main__":
    # Run ablation experiment
    ablation_results = ablation_experiment()