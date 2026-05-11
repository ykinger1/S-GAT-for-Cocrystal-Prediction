# ===================== 1. Import Dependencies =====================
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from model_Dual_branch_Graph_attention import DualBranchGraphAttentionModel
from smiles_to_graph import smiles_to_graph
from dataset_splitting import split_train_val_test
from evaluation import binary_classification_evaluation

# ===================== 2. Hyperparameter Configuration =====================
# Dataset configuration
DATASET = "2_SMILES"
DATA_PATH = "../data/dataset_" + DATASET + ".csv"
STRATIFY_COL = "cocrystal"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42
SMILES_COL1 = "SMILES1"  # Column name for first SMILES string
SMILES_COL2 = "SMILES2"  # Column name for second SMILES string
LABEL_COL = "cocrystal"  # Column name for target label

# Model configuration
ATOM_IN_CHANNELS = 39  # Fixed atomic feature dimension (matches smiles_to_graph output)
GAT_HIDDEN_CHANNELS = 256  # GAT hidden layer dimension
GAT_OUT_CHANNELS = 128  # GAT node-level output dimension
GAT_HEADS = 4  # Number of multi-head attention heads in GAT
GAT_DROPOUT = 0.1  # Dropout rate for GAT layers
READOUT = "sum"  # Graph pooling method: mean/sum
MLP_HIDDEN_DIM = 1024  # Hidden dimension of MLP classifier head
MLP_DROPOUT = 0.3  # Dropout rate for MLP layers

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
EPOCHS = 50
PATIENCE = 5  # Early stopping patience

# Saving configuration
MODEL_SAVE_DIR = "trained"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "dual_branch_graph_attention_" + DATASET + ".pth")
ROC_SAVE_DIR = "roc_curves"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Training device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔧 GPU name: {torch.cuda.get_device_name(0)}")


# ===================== Set Global Random Seed (Reproducibility) =====================
def set_global_seed(seed):
    """
    Fix random seeds across all libraries to ensure reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ===================== 3. Dataset Class (SMILES Processing) =====================
class SmilesGraphDataset(Dataset):
    """
    Dataset class for processing SMILES strings into graph data for dual-branch GAT model.

    Args:
        df (pd.DataFrame): Input dataframe containing SMILES and label columns
        smiles_col1 (str): Column name for first SMILES string
        smiles_col2 (str): Column name for second SMILES string
        label_col (str): Column name for target label
    """

    def __init__(self, df, smiles_col1, smiles_col2, label_col):
        self.smiles1_list = df[smiles_col1].values
        self.smiles2_list = df[smiles_col2].values
        self.labels = df[label_col].values.astype(np.float32)

    def __len__(self):
        """Return total number of samples in dataset"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get single sample by index.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (smiles1, smiles2, label)
        """
        return self.smiles1_list[idx], self.smiles2_list[idx], self.labels[idx]


# ===================== Custom Collate Function (SMILES→PyG Batch) =====================
def graph_collate_fn(batch):
    """
    Convert a batch of SMILES data into PyTorch Geometric Batch objects (for dual-branch GAT).

    Args:
        batch (list): Each element is a tuple (smiles1, smiles2, label)

    Returns:
        tuple: (batch1, batch2, labels) where:
            - batch1: PyG Batch for first molecule branch
            - batch2: PyG Batch for second molecule branch
            - labels: Tensor of target labels (float32)
    """
    smiles1_batch, smiles2_batch, labels_batch = [], [], []
    for s1, s2, label in batch:
        smiles1_batch.append(s1)
        smiles2_batch.append(s2)
        labels_batch.append(label)

    # SMILES → Graph list (convert to PyG Data) → PyG Batch
    def smiles_to_pyg_data(smiles):
        """Convert dictionary output from smiles_to_graph to PyG Data object"""
        graph_dict = smiles_to_graph(smiles, return_torch=True)
        return Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"]
        )

    graphs1 = [smiles_to_pyg_data(s) for s in smiles1_batch]
    graphs2 = [smiles_to_pyg_data(s) for s in smiles2_batch]
    batch1 = Batch.from_data_list(graphs1).to(DEVICE)
    batch2 = Batch.from_data_list(graphs2).to(DEVICE)
    labels = torch.tensor(labels_batch, dtype=torch.float32).to(DEVICE)

    return batch1, batch2, labels


# ===================== 4. Model Training Function =====================
def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device, patience):
    """
    Train dual-branch graph attention model with early stopping.

    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model (nn.Module): Dual-branch GAT model
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        epochs (int): Maximum training epochs
        device (torch.device): Training device (cuda/cpu)
        patience (int): Early stopping patience rounds

    Returns:
        nn.Module: Trained model with best validation loss
    """
    best_val_loss = float('inf')
    patience_counter = 0
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_idx, (batch1, batch2, labels) in enumerate(train_loader):
            # Forward pass (dual-branch GAT takes two molecule batches)
            outputs = model(batch1, batch2)
            loss = criterion(outputs, labels.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch1, batch2, labels in val_loader:
                outputs = model(batch1, batch2)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item() * len(labels)

        # Calculate average losses
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)

        print(f"📌 Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"⚠️ Early stopping triggered (no validation loss improvement for {patience} consecutive epochs) | Best Val Loss: {best_val_loss:.4f}")
                break

    # Load best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    return model


# ===================== 5. Prediction and Evaluation Function =====================
def predict_and_evaluate(model, dataloader, device, dataset_name):
    """
    Evaluate model performance on given dataset (train/val/test) with binary classification metrics.

    Args:
        model (nn.Module): Trained dual-branch GAT model
        dataloader (DataLoader): Data loader for evaluation
        device (torch.device): Inference device (cuda/cpu)
        dataset_name (str): Name of dataset (train/val/test) for logging

    Returns:
        tuple: (metrics_dict, auc_score) where:
            - metrics_dict: Dictionary of classification metrics (accuracy, precision, recall, f1)
            - auc_score: ROC-AUC score
    """
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for batch1, batch2, labels in dataloader:
            outputs = model(batch1, batch2)
            # Convert logits to probabilities (sigmoid)
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores >= 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.reshape(-1))
            y_score.extend(scores.reshape(-1))

    # Create ROC save directory if not exists
    os.makedirs(ROC_SAVE_DIR, exist_ok=True)
    roc_save_path = os.path.join(ROC_SAVE_DIR, f"roc_{dataset_name}_" + DATASET + ".png")

    # Calculate evaluation metrics
    metrics, auc = binary_classification_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        plot_roc=True,
        save_roc_path=roc_save_path
    )

    # Print evaluation results
    print(f"\n{'=' * 50}")
    print(f"📊 {dataset_name.capitalize()} Set Evaluation Results")
    print(f"{'=' * 50}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<12}: {metric_value:.4f}")
    print(f"AUC score    : {auc:.4f}")
    print(f"{'=' * 50}")
    return metrics, auc


# ===================== 6. Main Function =====================
def main():
    """Main pipeline: data loading → dataset splitting → model training → evaluation"""
    # Fix random seed for reproducibility
    set_global_seed(RANDOM_SEED)
    print(f"🔒 Global random seed fixed to: {RANDOM_SEED}")

    # Create save directories
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(ROC_SAVE_DIR, exist_ok=True)

    # 1. Load raw dataset
    print("\n📥 Loading raw dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw dataset shape: {df.shape} | Positive samples: {df[LABEL_COL].sum()} | Negative samples: {len(df) - df[LABEL_COL].sum()}")

    # 2. Stratified dataset splitting (train/val/test)
    print("\n🔪 Performing stratified train/validation/test split...")
    train_df, val_df, test_df = split_train_val_test(
        df=df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify_col=STRATIFY_COL,
        random_seed=RANDOM_SEED
    )
    print(f"Train set: {train_df.shape} | Validation set: {val_df.shape} | Test set: {test_df.shape}")

    # 3. Build datasets and data loaders
    print("\n📦 Building data loaders...")
    train_dataset = SmilesGraphDataset(train_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    val_dataset = SmilesGraphDataset(val_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    test_dataset = SmilesGraphDataset(test_df, SMILES_COL1, SMILES_COL2, LABEL_COL)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=graph_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=graph_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=graph_collate_fn
    )

    # 4. Initialize dual-branch graph attention model
    print("\n🔧 Initializing dual-branch graph attention model...")
    model = DualBranchGraphAttentionModel(
        atom_in_channels=ATOM_IN_CHANNELS,
        gat_hidden_channels=GAT_HIDDEN_CHANNELS,
        gat_out_channels=GAT_OUT_CHANNELS,
        gat_heads=GAT_HEADS,
        gat_dropout=GAT_DROPOUT,
        readout=READOUT,
        mlp_hidden_channels=MLP_HIDDEN_DIM,
        mlp_dropout=MLP_DROPOUT
    )

    # 5. Loss function and optimizer (handle class imbalance)
    train_labels = train_df[LABEL_COL].values
    pos_sample_num = sum(train_labels)
    neg_sample_num = len(train_labels) - pos_sample_num
    pos_weight = torch.tensor([neg_sample_num / pos_sample_num], device=DEVICE) if pos_sample_num > 0 else torch.tensor([1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # More stable with logits output
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Fix seed again before training (additional safeguard)
    set_global_seed(RANDOM_SEED)

    # 6. Train or load pre-trained model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n✅ Pre-trained model detected: {MODEL_SAVE_PATH} | Loading model weights...")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print("\n🚀 Starting model training...")
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=DEVICE,
            patience=PATIENCE
        )
        print(f"\n✅ Model training completed | Saved to: {MODEL_SAVE_PATH}")

    # 7. Evaluate model performance
    print("\n📈 Evaluating on training set...")
    train_metrics, train_auc = predict_and_evaluate(model, train_loader, DEVICE, "train")
    print("\n📈 Evaluating on validation set...")
    val_metrics, val_auc = predict_and_evaluate(model, val_loader, DEVICE, "val")
    print("\n📈 Evaluating on test set...")
    test_metrics, test_auc = predict_and_evaluate(model, test_loader, DEVICE, "test")


if __name__ == "__main__":
    # Check RDKit availability (required for SMILES processing)
    try:
        from rdkit import Chem

        assert Chem.MolFromSmiles("C") is not None
    except (ImportError, AssertionError):
        print("❌ RDKit is not properly installed. Please install RDKit first: conda install -c conda-forge rdkit")
        exit(1)

    # Check PyTorch Geometric availability
    try:
        import torch_geometric
    except ImportError:
        print("❌ PyTorch Geometric is not installed. Please install first: pip install torch_geometric")
        exit(1)

    # Run main pipeline
    main()