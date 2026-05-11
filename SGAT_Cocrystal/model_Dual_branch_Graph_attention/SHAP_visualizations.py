import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import os
from typing import List, Tuple, Optional

# ===================== Core Configuration: Adjust DPI/Resolution =====================
# Adjust resolution here: larger values mean higher DPI (clearer images but larger file size)
# Standard resolution: (400, 400) | High resolution: (800, 800) | Ultra-high resolution: (1200, 1200)
IMG_SIZE = (600, 600)

BONDLINEWIDTH = 6  # Bond line width (larger values = thicker bonds)
HIGHLIGHTRADIUS = 0.3  # Highlight radius for atoms (larger values = bigger highlight circles)

# Color palettes for atom highlighting
COLOR_LIST1 = [
    (245, 124, 110),
    (242, 181, 110),
    (132, 195, 183),
    (113, 184, 237),
    (184, 174, 234),
    (242, 168, 218)
]
COLOR_LIST2 = [
    (110, 124, 245),
    (110, 181, 242),
    (183, 195, 132),
    (237, 184, 113),
    (234, 174, 184),
    (218, 168, 242)
]
COLOR_LIST3 = [
    (245, 124, 110),
    (242, 181, 110),
    (132, 195, 183),
    (113, 184, 237),
    (184, 174, 234),
    (242, 168, 218)
]
COLOR_LIST4 = [
    (110, 124, 245),
    (110, 181, 242),
    (183, 195, 132),
    (237, 184, 113),
    (234, 174, 184),
    (218, 168, 242)
]


# ==================================================================

def rgb_list_to_rdkit(rgb_list: List[Tuple[int, int, int]]) -> List[Tuple[float, float, float]]:
    """
    Convert RGB values (0-255) to RDKit-compatible RGB tuples (0.0-1.0).

    Parameters:
        rgb_list: List of (R, G, B) tuples with integer values (0-255)

    Returns:
        List of (R, G, B) tuples with float values normalized to 0.0-1.0
    """
    return [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in rgb_list]


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing illegal characters and truncating length.

    Parameters:
        filename: Original filename to sanitize

    Returns:
        Sanitized filename with illegal characters replaced by underscores and length limited to 50 chars
    """
    illegal_chars = r'\/:*?"<>|'
    for char in illegal_chars:
        filename = filename.replace(char, "_")
    return filename[:50]


def visualize_molecule_atoms(
        smiles: str, atom_positions: List[Optional[int]], color_list: List[Tuple[float, float, float]],
        save_path: str, row_num: int, mol_type: str
) -> bool:
    """
    Generate a 2D visualization of a molecule with specified atoms highlighted in given colors.

    Parameters:
        smiles: SMILES string of the molecule to visualize
        atom_positions: List of 1-based atom indices to highlight (None/NaN for no highlight)
        color_list: List of RDKit-compatible RGB tuples (0.0-1.0) for atom highlighting
        save_path: Path to save the generated image
        row_num: Row number in the input CSV (for logging)
        mol_type: Type of molecule (e.g., "mol1", "mol2") for error logging

    Returns:
        True if visualization is generated and saved successfully; False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"[Row {row_num}] Error: Failed to parse {mol_type} SMILES string")
        return False

    # Atom highlight dictionary (maps 0-based atom index to highlight color)
    highlight_dict = {}
    for top_idx, pos in enumerate(atom_positions):
        if pd.isna(pos) or pos is None:
            continue
        try:
            atom_idx = int(pos) - 1  # Convert to 0-based index
            if 0 <= atom_idx < mol.GetNumAtoms():
                highlight_dict[atom_idx] = color_list[top_idx]
        except Exception:
            # Skip invalid atom positions
            continue

    try:
        # Initialize Cairo drawer with specified image size
        drawer = rdMolDraw2D.MolDraw2DCairo(IMG_SIZE[0], IMG_SIZE[1])
        opts = drawer.drawOptions()
        opts.bondLineWidth = BONDLINEWIDTH  # Set bond line width
        opts.highlightRadius = HIGHLIGHTRADIUS  # Set atom highlight radius

        # Draw molecule with atom highlights (no bond highlights)
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(highlight_dict.keys()),
            highlightAtomColors=highlight_dict,
            highlightBonds=[]
        )
        drawer.FinishDrawing()
        drawer.WriteDrawingText(save_path)
        print(f"[Row {row_num}] Successfully saved: {save_path}")
        return True
    except Exception as e:
        print(f"[Row {row_num}] Error: Failed to generate visualization - {str(e)}")
        return False


def process_csv(csv_path: str, output_dir: str = "mol_visualizations"):
    """
    Process a CSV file to generate molecular visualizations for all valid rows.

    Parameters:
        csv_path: Path to input CSV file containing SMILES, atom positions, and cocrystal labels
        output_dir: Directory to save generated molecular images (created if not exists)

    Notes:
        CSV must contain columns:
        - SMILES1: SMILES string for molecule 1
        - SMILES2: SMILES string for molecule 2
        - cocrystal: Binary label (0/1) for cocrystal status
        - mol1_top1_pos to mol1_top6_pos: 1-based indices of top 6 atoms for molecule 1
        - mol2_top1_pos to mol2_top6_pos: 1-based indices of top 6 atoms for molecule 2
    """
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Define columns for top atom positions of each molecule
    mol1_cols = [f"mol1_top{i}_pos" for i in range(1, 7)]
    mol2_cols = [f"mol2_top{i}_pos" for i in range(1, 7)]

    # Iterate over each row in the CSV (1-based row numbering)
    for row_num, (_, row) in enumerate(df.iterrows(), start=1):
        s1, s2, coc = str(row["SMILES1"]), str(row["SMILES2"]), row["cocrystal"]

        # Validate and parse cocrystal label (must be 0 or 1)
        try:
            coc_val = int(float(coc))
            if coc_val not in (0, 1):
                continue
        except Exception:
            print(f"[Row {row_num}] Error: Failed to parse cocrystal value (must be 0/1), skipping row")
            continue

        # Process molecule 1
        pos1 = [row[col] for col in mol1_cols]
        mol1_colors = rgb_list_to_rdkit(COLOR_LIST1 if coc_val == 1 else COLOR_LIST2)
        name1 = f"{row_num}_{coc_val}_{sanitize_filename(s1)}_{sanitize_filename(s2)}_1.png"
        visualize_molecule_atoms(
            smiles=s1,
            atom_positions=pos1,
            color_list=mol1_colors,
            save_path=os.path.join(output_dir, name1),
            row_num=row_num,
            mol_type="mol1"
        )

        # Process molecule 2
        pos2 = [row[col] for col in mol2_cols]
        mol2_colors = rgb_list_to_rdkit(COLOR_LIST3 if coc_val == 1 else COLOR_LIST4)
        name2 = f"{row_num}_{coc_val}_{sanitize_filename(s1)}_{sanitize_filename(s2)}_2.png"
        visualize_molecule_atoms(
            smiles=s2,
            atom_positions=pos2,
            color_list=mol2_colors,
            save_path=os.path.join(output_dir, name2),
            row_num=row_num,
            mol_type="mol2"
        )


if __name__ == "__main__":
    # Configuration for input/output paths
    INPUT_CSV = r"graphshap_final_results/graphshap_27cols_final.csv"  # Path to input CSV file
    OUTPUT_DIR = "mol_visualizations-600"  # Directory for output images

    # Run the main processing function
    process_csv(INPUT_CSV, OUTPUT_DIR)
    print("✅ All processing completed!")