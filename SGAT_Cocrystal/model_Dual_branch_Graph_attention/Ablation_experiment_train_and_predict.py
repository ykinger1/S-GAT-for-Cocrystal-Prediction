"""
Ablation Experiment Training and Prediction Script for Cocrystal Prediction
--------------------------------------------------------------------------
This script implements the complete training, validation, and testing pipeline for ablation experiments
on cocrystal formation prediction using graph neural networks (GNNs) and multi-layer perceptrons (MLPs).
It supports multiple ablation models, stratified dataset splitting, reproducible training,
and comprehensive binary classification evaluation (including AUC, accuracy, precision, recall, F1-score).

Dependencies:
- Python 3.8+
- PyTorch 1.10+ (with CUDA support recommended)
- PyTorch Geometric 2.0+
- RDKit 2022.09+ (for SMILES-to-graph conversion)
- NumPy 1.21+, Pandas 1.4+, Scikit-learn 1.0+

Usage:
    python Ablation_experiment_train_and_predict.py

Key Features:
1. Reproducible training via global random seed fixation
2. Stratified train/validation/test dataset splitting
3. Custom SMILES-to-graph dataset loader with PyTorch Geometric integration
4. Early stopping mechanism to prevent overfitting
5. Automatic model saving and result logging
6. Comprehensive evaluation metrics for binary classification

Author: [Your Name/Research Group]
Affiliation: [Your Institution]
Date: [YYYY-MM-DD]
"""

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

# Import all ablation experiment models
from Ablation_experiment_model import (
    BaselineModel, A1Model, A2Model, A3Model,
    A4Model, A5Model, A6Model, A7Model, A8Model
)
from smiles_to_graph import smiles_to_graph  # SMILES to graph conversion function
from dataset_splitting import split_train_val_test  # Dataset splitting utility
from evaluation import binary_classification_evaluation  # Evaluation metrics

# ===================== 2. Hyperparameter Configuration =====================
# Data-related configuration
DATASET = "2_SMILES"
DATA_PATH = "../data/dataset_" + DATASET + ".csv"
STRATIFY_COL = "cocrystal"
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42
SMILES_COL1 = "SMILES1"
SMILES_COL2 = "SMILES2"
LABEL_COL = "cocrystal"

# Model general configuration
ATOM_IN_CHANNELS = 39
GAT_HIDDEN_CHANNELS = 256
GAT_OUT_CHANNELS = 128
GAT_HEADS = 4
GAT_DROPOUT = 0.1
READOUT = "sum"
MLP_HIDDEN_DIM = 1024
MLP_DROPOUT = 0.3

# Training-related configuration
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-8
EPOCHS = 50
PATIENCE = 5  # Early stopping patience

# Saving-related configuration
BASE_SAVE_DIR = "ablation_experiments"
MODEL_SAVE_ROOT = os.path.join(BASE_SAVE_DIR, "trained_models")
ROC_SAVE_ROOT = os.path.join(BASE_SAVE_DIR, "roc_curves")
RESULT_SAVE_PATH = os.path.join(BASE_SAVE_DIR, "ablation_results.csv")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Training Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"🔧 GPU Name: {torch.cuda.get_device_name(0)}")

# Define all ablation experiment configurations
ABLATION_MODELS = {
    "A1": {
        "class": A1Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gcn_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gcn_out_channels": GAT_OUT_CHANNELS,
            "gcn_dropout": GAT_DROPOUT,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A2": {
        "class": A2Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A3": {
        "class": A3Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "atom_mlp_hidden": GAT_OUT_CHANNELS,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A4": {
        "class": A4Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gat_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gat_out_channels": GAT_OUT_CHANNELS,
            "gat_heads": GAT_HEADS,
            "gat_dropout": GAT_DROPOUT,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A5": {
        "class": A5Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gat_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gat_out_channels": GAT_OUT_CHANNELS,
            "gat_heads": GAT_HEADS,
            "gat_dropout": GAT_DROPOUT,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A6": {
        "class": A6Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gat_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gat_out_channels": GAT_OUT_CHANNELS,
            "gat_heads": GAT_HEADS,
            "gat_dropout": GAT_DROPOUT,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A7": {
        "class": A7Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gat_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gat_out_channels": GAT_OUT_CHANNELS,
            "gat_heads": GAT_HEADS,
            "gat_dropout": GAT_DROPOUT,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM,
            "mlp_dropout": MLP_DROPOUT
        }
    },
    "A8": {
        "class": A8Model,
        "params": {
            "atom_in_channels": ATOM_IN_CHANNELS,
            "gat_hidden_channels": GAT_HIDDEN_CHANNELS,
            "gat_out_channels": GAT_OUT_CHANNELS,
            "gat_heads": GAT_HEADS,
            "readout": READOUT,
            "mlp_hidden_channels": MLP_HIDDEN_DIM
        }
    }
}


# ===================== Global Random Seed Fixation =====================
def set_global_seed(seed):
    """
    Fix all random seeds to ensure full reproducibility of experiments.

    Args:
        seed (int): Random seed value
    """
    # Python built-in random number generator
    random.seed(seed)
    # NumPy random number generator
    np.random.seed(seed)
    # PyTorch CPU/GPU random number generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # CuDNN deterministic mode (disable benchmark for reproducibility)
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Global random seed fixed to: {seed}")


# ===================== 3. Dataset Class =====================
class SmilesGraphDataset(Dataset):
    """
    Custom Dataset class for loading SMILES strings and converting them to graph data.

    Args:
        df (pd.DataFrame): Input DataFrame containing SMILES and label columns
        smiles_col1 (str): Column name for first SMILES string
        smiles_col2 (str): Column name for second SMILES string
        label_col (str): Column name for binary label (cocrystal formation)
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
        Get a single sample from dataset by index.

        Args:
            idx (int): Sample index

        Returns:
            tuple: (smiles1, smiles2, label)
        """
        return self.smiles1_list[idx], self.smiles2_list[idx], self.labels[idx]


# ===================== Custom Collate Function =====================
def graph_collate_fn(batch):
    """
    Custom collate function for batching graph data from SMILES strings.

    Args:
        batch (list): List of tuples (smiles1, smiles2, label)

    Returns:
        tuple: (batch1, batch2, labels) where batch1/batch2 are PyTorch Geometric Batch objects
    """
    smiles1_batch, smiles2_batch, labels_batch = [], [], []
    for s1, s2, label in batch:
        smiles1_batch.append(s1)
        smiles2_batch.append(s2)
        labels_batch.append(label)

    def smiles_to_pyg_data(smiles):
        """Convert SMILES string to PyTorch Geometric Data object"""
        graph_dict = smiles_to_graph(smiles, return_torch=True)
        return Data(
            x=graph_dict["x"],
            edge_index=graph_dict["edge_index"],
            edge_attr=graph_dict["edge_attr"]
        )

    # Convert SMILES batches to graph batches
    graphs1 = [smiles_to_pyg_data(s) for s in smiles1_batch]
    graphs2 = [smiles_to_pyg_data(s) for s in smiles2_batch]

    # Create PyTorch Geometric batches and move to device
    batch1 = Batch.from_data_list(graphs1).to(DEVICE)
    batch2 = Batch.from_data_list(graphs2).to(DEVICE)
    labels = torch.tensor(labels_batch, dtype=torch.float32).to(DEVICE)

    return batch1, batch2, labels


# ===================== 4. Model Training Function =====================
def train_model(train_loader, val_loader, model, criterion, optimizer, epochs, device, patience, model_save_path):
    """
    Train the model with early stopping based on validation loss.

    Args:
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        model (nn.Module): Model to train
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        epochs (int): Maximum number of training epochs
        device (torch.device): Training device (CPU/GPU)
        patience (int): Early stopping patience (number of epochs)
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
        for batch1, batch2, labels in train_loader:
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
        print(f"   Epoch [{epoch + 1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⚠️ Early stopping triggered! Best validation loss: {best_val_loss:.4f}")
                break

    # Load the best model weights
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return model


# ===================== 5. Prediction and Evaluation Function =====================
def predict_and_evaluate(model, dataloader, device, dataset_name, roc_save_path):
    """
    Generate predictions and evaluate model performance on a dataset.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Data loader for evaluation
        device (torch.device): Evaluation device
        dataset_name (str): Name of dataset (for logging)
        roc_save_path (str): Path to save ROC curve (if enabled)

    Returns:
        dict: Dictionary of evaluation metrics (accuracy, precision, recall, F1, AUC)
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

    # Calculate evaluation metrics
    metrics, auc = binary_classification_evaluation(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        plot_roc=False,
        save_roc_path=roc_save_path
    )

    # Print evaluation results
    print(f"\n   {'=' * 40}")
    print(f"   📊 {dataset_name} Set Evaluation Results")
    print(f"   {'=' * 40}")
    for k, v in metrics.items():
        print(f"   {k:<12}: {v:.4f}")
    print(f"   AUC         : {auc:.4f}")
    print(f"   {'=' * 40}")

    # Package results
    result = {f"{dataset_name}_{k}": v for k, v in metrics.items()}
    result[f"{dataset_name}_auc"] = auc
    return result


# ===================== 6. Single Ablation Experiment =====================
def run_single_ablation(model_name, model_config, data_loaders, save_dirs):
    """
    Run training and evaluation for a single ablation model.

    Args:
        model_name (str): Name of ablation model (e.g., "A1", "A2")
        model_config (dict): Dictionary containing model class and parameters
        data_loaders (dict): Dictionary of train/val/test data loaders
        save_dirs (dict): Dictionary of save directories (model, roc)

    Returns:
        dict: Results of the ablation experiment
    """
    print(f"\n{'=' * 60}")
    print(f"Starting Training/Evaluation for Ablation Model: {model_name}")
    print(f"{'=' * 60}")

    # Initialize model
    model = model_config["class"](**model_config["params"])

    # Path configuration
    model_save_path = os.path.join(save_dirs["model"], f"{model_name}_{DATASET}.pth")
    roc_test_path = os.path.join(save_dirs["roc"], f"roc_{model_name}_test_{DATASET}.png")

    # Loss function and optimizer (with class weight balancing)
    train_labels = data_loaders["train_dataset"].labels
    pos_num = sum(train_labels)
    neg_num = len(train_labels) - pos_num
    pos_weight = torch.tensor([neg_num / pos_num], device=DEVICE) if pos_num > 0 else torch.tensor([1.0], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Fix random seed for reproducibility
    set_global_seed(RANDOM_SEED)

    # Train or load pre-trained model
    if os.path.exists(model_save_path):
        print(f"   ✅ Loading pre-trained model: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print(f"   🚀 Starting model training...")
        model = train_model(
            train_loader=data_loaders["train"],
            val_loader=data_loaders["val"],
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epochs=EPOCHS,
            device=DEVICE,
            patience=PATIENCE,
            model_save_path=model_save_path
        )
        print(f"   ✅ Model training completed!")

    # Evaluate on test set (core focus of ablation experiments)
    print(f"\n   📈 Evaluating on test set...")
    test_result = predict_and_evaluate(model, data_loaders["test"], DEVICE, "test", roc_test_path)

    # Aggregate results
    return {"model_name": model_name, **test_result}


# ===================== 7. Main Function =====================
def main():
    """Main function to execute ablation experiments pipeline"""
    # Create save directories if not exist
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)
    os.makedirs(ROC_SAVE_ROOT, exist_ok=True)

    # 1. Load dataset
    print("\n📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape} | Positive samples: {df[LABEL_COL].sum()} | Negative samples: {len(df) - df[LABEL_COL].sum()}")

    # 2. Stratified dataset splitting
    print("\n🔪 Splitting train/validation/test sets (stratified)...")
    train_df, val_df, test_df = split_train_val_test(
        df=df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify_col=STRATIFY_COL,
        random_seed=RANDOM_SEED
    )
    print(f"Train set: {train_df.shape} | Val set: {val_df.shape} | Test set: {test_df.shape}")

    # 3. Build data loaders
    train_dataset = SmilesGraphDataset(train_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    val_dataset = SmilesGraphDataset(val_df, SMILES_COL1, SMILES_COL2, LABEL_COL)
    test_dataset = SmilesGraphDataset(test_df, SMILES_COL1, SMILES_COL2, LABEL_COL)

    train_loader = DataLoader(
        train_dataset,
        BATCH_SIZE,
        shuffle=True,
        collate_fn=graph_collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        BATCH_SIZE,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        BATCH_SIZE,
        shuffle=False,
        collate_fn=graph_collate_fn,
        num_workers=0
    )

    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset
    }
    save_dirs = {"model": MODEL_SAVE_ROOT, "roc": ROC_SAVE_ROOT}

    # 4. Run ablation experiments in batch
    all_results = []
    for model_name, config in ABLATION_MODELS.items():
        res = run_single_ablation(model_name, config, data_loaders, save_dirs)
        all_results.append(res)

    # 5. Save and display results
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(RESULT_SAVE_PATH, index=False, encoding="utf-8")
    print(f"\n📝 All ablation experiment results saved to: {RESULT_SAVE_PATH}")
    print("\n📋 Ablation Experiment Summary:")
    print(result_df)


if __name__ == "__main__":
    # Dependency check
    try:
        from rdkit import Chem

        assert Chem.MolFromSmiles("C") is not None, "RDKit failed to parse SMILES"
    except ImportError:
        print("❌ RDKit is not installed. Please install it via: conda install -c conda-forge rdkit")
        exit(1)
    except AssertionError:
        print("❌ RDKit is installed but not functioning correctly")
        exit(1)

    try:
        import torch_geometric
    except ImportError:
        print("❌ PyTorch Geometric is not installed. Please install it via: pip install torch_geometric")
        exit(1)

    # Execute main pipeline
    main()