# ===================== 1. Import Dependencies =====================
import pandas as pd
from sklearn.model_selection import train_test_split


# ===================== 2. Function Definition =====================
def split_train_val_test(df, test_size=0.2, val_size=0.1, stratify_col="cocrystal", random_seed=42):
    """
    Split the dataset into mutually exclusive training, validation, and test sets using stratified sampling.

    Logic:
        1. First split the test set from the original dataset (stratified sampling)
        2. Then split the remaining data into training and validation sets (stratified sampling)
        This ensures all three sets are stratified and non-overlapping.

    Parameters:
        df (pd.DataFrame): Original dataset in DataFrame format
        test_size (float): Proportion of the original dataset allocated to the test set (0-1)
        val_size (float): Proportion of the original dataset allocated to the validation set (0-1)
        stratify_col (str): Column name used for stratified sampling (to maintain class distribution)
        random_seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Training set DataFrame
        pd.DataFrame: Validation set DataFrame
        pd.DataFrame: Test set DataFrame
    """

    # Step 1: Split test set and temporary (train+validation) set (stratified sampling)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df[stratify_col]
    )

    # Step 2: Split training and validation sets from the temporary set (stratified sampling)
    # Calculate validation ratio relative to the temporary set (to ensure val_size of original dataset)
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=train_val_df[stratify_col]
    )

    return train_df, val_df, test_df


# ===================== 3. Main Function (Test Code) =====================
if __name__ == "__main__":
    # ---------------------- Configurable Parameters (Modify as Needed) ----------------------
    CSV_FILE_PATH = "data/dataset_2_SMILES.csv"  # Path to original dataset
    TEST_SIZE = 0.2  # Test set proportion (20% of total)
    VAL_SIZE = 0.1  # Validation set proportion (10% of total), training set = 70%
    RANDOM_SEED = 42  # Random seed for reproducibility
    STRATIFY_COL = "cocrystal"  # Column for stratified sampling (no modification required)

    # ---------------------- Data Loading and Preprocessing ----------------------
    # Load original dataset
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"📊 Original dataset shape: {df.shape}")
    print(f"📌 First 2 rows of original dataset:\n{df.head(2)}\n")

    # ---------------------- Stratified Dataset Splitting ----------------------
    train_df, val_df, test_df = split_train_val_test(
        df,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        stratify_col=STRATIFY_COL,
        random_seed=RANDOM_SEED
    )

    # ---------------------- Result Validation ----------------------
    print("=" * 60)
    # Sample count validation
    total_samples = df.shape[0]
    train_ratio = train_df.shape[0] / total_samples
    val_ratio = val_df.shape[0] / total_samples
    test_ratio = test_df.shape[0] / total_samples
    print(f"📈 Sample distribution:")
    print(f"   Training set: {train_df.shape[0]} samples (proportion: {train_ratio:.4f})")
    print(f"   Validation set: {val_df.shape[0]} samples (proportion: {val_ratio:.4f})")
    print(f"   Test set: {test_df.shape[0]} samples (proportion: {test_ratio:.4f})")

    # Stratified column distribution validation (ensure consistent class distribution)
    print(f"\n🎯 {STRATIFY_COL} class distribution (percentage):")
    print(f"Original dataset:\n{df[STRATIFY_COL].value_counts(normalize=True).round(4) * 100}%")
    print(f"Training set:\n{train_df[STRATIFY_COL].value_counts(normalize=True).round(4) * 100}%")
    print(f"Validation set:\n{val_df[STRATIFY_COL].value_counts(normalize=True).round(4) * 100}%")
    print(f"Test set:\n{test_df[STRATIFY_COL].value_counts(normalize=True).round(4) * 100}%")

    # Independence validation (ensure no overlapping samples)
    train_ids = set(train_df.index)
    val_ids = set(val_df.index)
    test_ids = set(test_df.index)
    assert len(train_ids & val_ids) == 0, "❌ Overlap between training and validation sets!"
    assert len(train_ids & test_ids) == 0, "❌ Overlap between training and test sets!"
    assert len(val_ids & test_ids) == 0, "❌ Overlap between validation and test sets!"
    print("\n✅ Set independence validation passed: No overlap between training/validation/test sets!")
    print("\n🎉 Data processing completed! Training, validation, and test sets are ready for use.")