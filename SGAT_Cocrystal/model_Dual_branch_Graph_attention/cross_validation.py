# ==============================================================================
# Cross-Validation Implementation for Dual-Branch Graph Attention Network
# Paper: [Replace with your paper title]
# Author: [Replace with author information]
# Version: 1.0
# Description: K-fold cross-validation for cocrystal prediction using dual-branch GAT
# ==============================================================================
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from model_Dual_branch_Graph_attention import DualBranchGraphAttentionModel
from smiles_to_graph import smiles_to_graph  # Import SMILES to graph conversion function
from evaluation import binary_classification_evaluation

# ==============================================================================
# 1. Hyperparameter Configuration
# ==============================================================================
# Basic Configuration
DATASET = "2_SMILES"
DATA_PATH = "../data/dataset_" + DATASET + ".csv"
STRATIFY_COL = "cocrystal"
VAL_SIZE = 0.1  # Train-validation split ratio
RANDOM_SEED = 42
SMILES_COL1 = "SMILES1"  # Column name for SMILES 1
SMILES_COL2 = "SMILES2"  # Column name for SMILES 2
LABEL_COL = "cocrystal"

# Cross-Validation Configuration
K_FOLDS_LIST = [3, 5, 10]  # K values for K-fold cross-validation
REPEAT_TIMES = 1  # Optional: Number of repetitions for cross-validation (default: 1)

# Dual-Branch GAT Model Configuration
ATOM_IN_CHANNELS = 39  # Fixed atomic feature dimension
GAT_HIDDEN_CHANNELS = 256  # GAT hidden layer dimension
GAT_OUT_CHANNELS = 128  # GAT node-level output dimension
GAT_HEADS = 4  # Number of multi-head attention heads in GAT
GAT_DROPOUT = 0.1  # Dropout rate for GAT layers
READOUT = "sum"  # Graph pooling method: mean/sum
MLP_HIDDEN_DIM = 1024  # Hidden dimension of MLP classifier head
MLP_DROPOUT = 0.3  # Dropout rate for MLP layers

# Training Configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
EPOCHS = 50
PATIENCE = 5  # Early stopping patience

# Saving Configuration
MODEL_SAVE_ROOT = "trained_cv"
ROC_SAVE_ROOT = "roc_curves_cv"

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Training Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔧 GPU Name: {torch.cuda.get_device_name(0)}")


# ==============================================================================
# 2. Global Seed Fixing Function
# ==============================================================================
def set_global_seed(seed):
    """
    Fix random seeds for reproducibility across all libraries.

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


# ==============================================================================
# 3. Dataset Class (SMILES Processing)
# ==============================================================================
class SmilesGraphDataset(Dataset):
    """
    Dataset class for processing SMILES strings and corresponding labels.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES and labels
        smiles_col1 (str): Column name for first SMILES string
        smiles_col2 (str): Column name for second SMILES string
        label_col (str): Column name for target label
    """

    def __init__(self, df, smiles_col1, smiles_col2, label_col):
        self.smiles1_list = df[smiles_col1].values
        self.smiles2_list = df[smiles_col2].values
        self.labels = df[label_col].values.astype(np.float32)

    def __len__(self):
        """Return total number of samples in dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Get a single sample by index.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (smiles1, smiles2, label)
        """
        return self.smiles1_list[idx], self.smiles2_list[idx], self.labels[idx]


# ==============================================================================
# 4. Custom Collate Function (SMILES → PyG Batch)
# ==============================================================================
def graph_collate_fn(batch):
    """
    Convert a batch of SMILES data to PyTorch Geometric Batch objects (for dual-branch GAT).

    Args:
        batch (list): List of tuples (smiles1, smiles2, label)

    Returns:
        tuple: (batch1, batch2, labels) where:
            - batch1: PyG Batch for first SMILES branch
            - batch2: PyG Batch for second SMILES branch
            - labels: Tensor of target labels
    """
    smiles1_batch, smiles2_batch, labels_batch = [], [], []
    for s1, s2, label in batch:
        smiles1_batch.append(s1)
        smiles2_batch.append(s2)
        labels_batch.append(label)

    # Convert SMILES to PyG Data objects
    def smiles_to_pyg_data(smiles):
        graph_dict = smiles_to_graph(smiles, return_torch=True)
        return Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"]
        )

    # Convert SMILES lists to PyG Batch objects
    graphs1 = [smiles_to_pyg_data(s) for s in smiles1_batch]
    graphs2 = [smiles_to_pyg_data(s) for s in smiles2_batch]
    batch1 = Batch.from_data_list(graphs1).to(DEVICE)
    batch2 = Batch.from_data_list(graphs2).to(DEVICE)
    labels = torch.tensor(labels_batch, dtype=torch.float32).to(DEVICE)

    return batch1, batch2, labels


# ==============================================================================
# 5. Model Training Function
# ==============================================================================
def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device, patience, model_save_path):
    """
    Train the dual-branch GAT model with early stopping.

    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model (nn.Module): Dual-branch GAT model
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        epochs (int): Maximum training epochs
        device (torch.device): Training device (cuda/cpu)
        patience (int): Early stopping patience
        model_save_path (str): Path to save the best model

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
            outputs = model(batch1, batch2)
            loss = criterion(outputs, labels.unsqueeze(1))

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

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model


# ==============================================================================
# 6. Prediction and Evaluation Function
# ==============================================================================
def predict_and_evaluate(model, dataloader, device, save_roc_path=None):
    """
    Generate predictions and compute evaluation metrics for binary classification.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader for evaluation
        device (torch.device): Inference device
        save_roc_path (str, optional): Path to save ROC curve plot. Defaults to None.

    Returns:
        dict: Dictionary of evaluation metrics including AUC
    """
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for batch1, batch2, labels in dataloader:
            outputs = model(batch1, batch2)
            scores = torch.sigmoid(outputs).cpu().numpy()
            preds = (scores >= 0.5).astype(int)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.reshape(-1))
            y_score.extend(scores.reshape(-1))

    # Compute evaluation metrics
    metrics, auc = binary_classification_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        plot_roc=False,
        save_roc_path=save_roc_path
    )
    metrics['AUC'] = auc
    return metrics


# ==============================================================================
# 7. Single Fold Training and Evaluation Function
# ==============================================================================
def run_single_fold(fold_idx, k, train_val_df, test_df, seed):
    """
    Execute training, validation and testing for a single fold of cross-validation.

    Args:
        fold_idx (int): Index of current fold (0-based)
        k (int): Total number of folds
        train_val_df (pd.DataFrame): Training + validation DataFrame
        test_df (pd.DataFrame): Test DataFrame
        seed (int): Random seed for reproducibility

    Returns:
        dict: Evaluation metrics for train/val/test sets
    """
    # 1. Split train and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=VAL_SIZE,
        stratify=train_val_df[STRATIFY_COL],
        random_state=seed
    )

    # 2. Create data loaders
    train_dataset = SmilesGraphDataset(train_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    val_dataset = SmilesGraphDataset(val_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    test_dataset = SmilesGraphDataset(test_df, SMILES_COL1, SMILES_COL2, LABEL_COL)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=graph_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=graph_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=graph_collate_fn
    )

    # 3. Model saving path
    model_save_dir = os.path.join(MODEL_SAVE_ROOT, f"k{k}")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, f"dual_branch_gat_{DATASET}_fold{fold_idx}.pth")

    # 4. Initialize model
    set_global_seed(seed)  # Fix seed before model initialization
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

    # 5. Loss function and optimizer
    train_labels = train_df[LABEL_COL].values
    pos_sample_num = sum(train_labels)
    neg_sample_num = len(train_labels) - pos_sample_num
    pos_weight = torch.tensor([neg_sample_num / pos_sample_num], device=DEVICE) if pos_sample_num > 0 else torch.tensor([1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 6. Train or load pre-trained model
    if os.path.exists(model_save_path):
        print(f"✅ Pre-trained model detected: {model_save_path}, loading model...")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print(f"\n🚀 Starting training for Fold {fold_idx} of {k}-fold cross-validation...")
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=DEVICE,
            patience=PATIENCE,
            model_save_path=model_save_path
        )
        print(f"✅ Fold {fold_idx} training completed, model saved to: {model_save_path}")

    # 7. ROC curve saving path
    roc_save_dir = os.path.join(ROC_SAVE_ROOT, f"k{k}")
    os.makedirs(roc_save_dir, exist_ok=True)

    # 8. Evaluate model
    print(f"\n📈 Evaluating Fold {fold_idx} of {k}-fold cross-validation...")
    train_metrics = predict_and_evaluate(
        model, train_loader, DEVICE,
        save_roc_path=os.path.join(roc_save_dir, f"roc_train_{DATASET}_fold{fold_idx}.png")
    )
    val_metrics = predict_and_evaluate(
        model, val_loader, DEVICE,
        save_roc_path=os.path.join(roc_save_dir, f"roc_val_{DATASET}_fold{fold_idx}.png")
    )
    test_metrics = predict_and_evaluate(
        model, test_loader, DEVICE,
        save_roc_path=os.path.join(roc_save_dir, f"roc_test_{DATASET}_fold{fold_idx}.png")
    )

    return {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }


# ==============================================================================
# 8. K-Fold Cross-Validation Main Logic
# ==============================================================================
def run_kfold_cv():
    """Main function to execute K-fold cross-validation across specified K values."""
    # Load raw dataset
    print("\n📥 Loading raw dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw dataset shape: {df.shape} | Positive samples: {df[LABEL_COL].sum()} | Negative samples: {len(df) - df[LABEL_COL].sum()}")

    # Iterate over each K value
    for k in K_FOLDS_LIST:
        print(f"\n{'=' * 60}")
        print(f"Starting {k}-fold cross-validation")
        print(f"{'=' * 60}")

        # Store metrics across all folds
        all_fold_metrics = {
            'train': [],
            'val': [],
            'test': []
        }

        # Stratified K-fold splitting
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        fold_idx = 0
        for train_val_idx, test_idx in skf.split(df, df[STRATIFY_COL]):
            # Split train+validation and test sets
            train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)

            # Execute single fold training and evaluation
            fold_seed = RANDOM_SEED + fold_idx  # Unique seed for each fold
            fold_metrics = run_single_fold(fold_idx, k, train_val_df, test_df, fold_seed)

            # Collect metrics
            all_fold_metrics['train'].append(fold_metrics['train'])
            all_fold_metrics['val'].append(fold_metrics['val'])
            all_fold_metrics['test'].append(fold_metrics['test'])

            fold_idx += 1

        # Calculate mean and standard deviation of metrics
        print(f"\n{'=' * 60}")
        print(f"{k}-fold Cross-Validation Results Summary (Mean ± Std)")
        print(f"{'=' * 60}")

        # Define metrics to display
        metrics_list = ['AUC', 'Accuracy', 'BACC', 'Precision', 'Recall', 'Specificity', 'F1 score']

        for set_name in ['train', 'val', 'test']:
            print(f"\n📊 {set_name.upper()} Set Metrics:")
            for metric in metrics_list:
                # Extract metric values across all folds
                metric_values = [fold[metric] for fold in all_fold_metrics[set_name]]
                # Calculate mean and standard deviation
                mean_val = np.mean(metric_values)
                std_val = np.std(metric_values)
                # Print with 4 decimal places
                print(f"{metric:<12}: {mean_val:.4f} ± {std_val:.4f}")


# ==============================================================================
# 9. Main Entry Point
# ==============================================================================
def main():
    """Main entry point for the cross-validation pipeline."""
    # Check dependencies
    try:
        from rdkit import Chem
        assert Chem.MolFromSmiles("C") is not None
    except (ImportError, AssertionError):
        print("❌ RDKit is not properly installed. Please install RDKit first: conda install -c conda-forge rdkit")
        exit(1)

    try:
        import torch_geometric
    except ImportError:
        print("❌ PyTorch Geometric is not installed. Please install it first: pip install torch_geometric")
        exit(1)

    # Execute K-fold cross-validation
    run_kfold_cv()


if __name__ == "__main__":
    main()