from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np
import torch

# ===================== Feature Constant Definitions (Strictly match Table 1 in the reference paper) =====================
# Allowed values for atomic features
ATOM_SYMBOLS = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'metal']
ATOM_DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
    rdchem.HybridizationType.OTHER
]
NUM_HS = [0, 1, 2, 3, 4]
CHIRAL_TYPES = ['R', 'S']

# Allowed values for bond features
BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC
]
BOND_STEREO_TYPES = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOANY,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOE
]


# ===================== Helper Functions =====================
def one_hot_encoding(value, allowable_set):
    """
    General one-hot encoding function, matching the feature encoding rules in the reference paper.

    Parameters:
        value: Target value to be encoded
        allowable_set: List of valid values for the feature

    Returns:
        np.array: One-hot encoded vector
    """
    if value not in allowable_set:
        value = allowable_set[-1]  # Map invalid values to 'other/metal' category
    return np.array([1 if v == value else 0 for v in allowable_set], dtype=np.float32)


# ===================== Core Conversion Function =====================
def smiles_to_graph(smiles, return_torch=True):
    """
    Convert SMILES string to molecular graph structure required by Attentive FP.

    Parameters:
        smiles (str): SMILES representation of the molecule
        return_torch (bool): If True, return torch tensors; if False, return numpy arrays

    Returns:
        dict: Core molecular graph data with the following keys:
            x: Atomic feature matrix with shape (num_atoms, 39)
            edge_index: Edge index matrix with shape (2, num_edges) (undirected graph format)
            edge_attr: Bond feature matrix with shape (num_edges, 10)
            num_atoms: Number of heavy atoms in the molecule
            num_bonds: Number of chemical bonds in the molecule

    Raises:
        ValueError: If the input SMILES string is invalid and cannot be parsed
    """
    # 1. Convert SMILES to RDKit molecule object and validate
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string, cannot be parsed: {smiles}")

    # Assign stereochemistry information (chiral R/S, double bond E/Z configuration)
    # to match stereochemical feature encoding in the reference paper
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)

    # 2. Extract atomic features (39 dimensions total, strictly match Table 1 in the reference paper)
    num_atoms = mol.GetNumAtoms()
    atom_feature_list = []
    for atom in mol.GetAtoms():
        # 2.1 Atomic symbol (16-dimensional one-hot)
        atom_symbol = atom.GetSymbol()
        symbol_feat = one_hot_encoding(atom_symbol if atom_symbol in ATOM_SYMBOLS else 'metal', ATOM_SYMBOLS)

        # 2.2 Atomic degree (6-dimensional one-hot)
        degree_feat = one_hot_encoding(atom.GetDegree(), ATOM_DEGREES)

        # 2.3 Formal charge (1-dimensional numerical)
        formal_charge_feat = np.array([atom.GetFormalCharge()], dtype=np.float32)

        # 2.4 Number of radical electrons (1-dimensional numerical)
        radical_electron_feat = np.array([atom.GetNumRadicalElectrons()], dtype=np.float32)

        # 2.5 Hybridization type (6-dimensional one-hot)
        hybridization_feat = one_hot_encoding(atom.GetHybridization(), HYBRIDIZATIONS)

        # 2.6 Aromaticity flag (1-dimensional 0/1)
        aromatic_feat = np.array([1 if atom.GetIsAromatic() else 0], dtype=np.float32)

        # 2.7 Total number of bonded hydrogen atoms (5-dimensional one-hot)
        num_hs_feat = one_hot_encoding(atom.GetTotalNumHs(), NUM_HS)

        # 2.8 Chiral center flag (1-dimensional 0/1)
        is_chiral_feat = np.array([1 if atom.HasProp('_CIPCode') else 0], dtype=np.float32)

        # 2.9 Chiral type (R/S) (2-dimensional one-hot)
        chiral_type = atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else ''
        chiral_type_feat = one_hot_encoding(chiral_type, CHIRAL_TYPES)

        # Concatenate full atomic features
        full_atom_feat = np.concatenate([
            symbol_feat, degree_feat, formal_charge_feat, radical_electron_feat,
            hybridization_feat, aromatic_feat, num_hs_feat, is_chiral_feat, chiral_type_feat
        ])
        atom_feature_list.append(full_atom_feat)

    # Convert to atomic feature matrix
    atom_features = np.array(atom_feature_list, dtype=np.float32)

    # 3. Extract edge indices and bond features (undirected graph with bidirectional edges,
    # matching GNN input specifications)
    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()

        # 3.1 Bond type (4-dimensional one-hot)
        bond_type_feat = one_hot_encoding(bond.GetBondType(), BOND_TYPES)

        # 3.2 Conjugation flag (1-dimensional 0/1)
        conjugation_feat = np.array([1 if bond.GetIsConjugated() else 0], dtype=np.float32)

        # 3.3 Ring membership flag (1-dimensional 0/1)
        in_ring_feat = np.array([1 if bond.IsInRing() else 0], dtype=np.float32)

        # 3.4 Bond stereoconfiguration (E/Z) (4-dimensional one-hot)
        bond_stereo_feat = one_hot_encoding(bond.GetStereo(), BOND_STEREO_TYPES)

        # Concatenate full bond features
        full_bond_feat = np.concatenate([
            bond_type_feat, conjugation_feat, in_ring_feat, bond_stereo_feat
        ])

        # Add bidirectional edges for undirected graph
        edge_index_list.append([u, v])
        edge_attr_list.append(full_bond_feat)
        edge_index_list.append([v, u])
        edge_attr_list.append(full_bond_feat)

    # Convert to edge index and bond feature matrices
    edge_index = np.array(edge_index_list, dtype=np.int64).T  # Transpose to GNN standard (2, num_edges) format
    edge_attr = np.array(edge_attr_list, dtype=np.float32)
    num_bonds = mol.GetNumBonds()

    # 4. Optional conversion to PyTorch tensors
    if return_torch:
        atom_features = torch.tensor(atom_features, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    # 5. Return complete molecular graph data
    return {
        "x": atom_features,  # Atomic feature matrix
        "edge_index": edge_index,  # Edge index matrix
        "edge_attr": edge_attr,  # Bond feature matrix
        "num_atoms": num_atoms,  # Number of atoms in the molecule
        "num_bonds": num_bonds  # Number of bonds in the molecule
    }


# ===================== Test Main Function =====================
def main():
    """
    Test the smiles_to_graph function with typical molecules mentioned in the reference paper,
    including aspirin, iprodione, methanol, methane, benzene, and chiral molecules (L-Alanine).
    """
    # Test cases (core example molecules from the reference paper)
    test_molecules = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Iprodione": "CC(C)NC(=O)N1C(=O)C(NC(=O)NC2=CC=C(Cl)C=C2Cl)C(=O)N1",
        "Methanol": "CO",
        "Methane": "C",
        "Benzene": "c1ccccc1",
        "Chiral molecule (L-Alanine)": "C[C@H](N)C(=O)O"
    }

    # Batch testing
    for mol_name, smiles in test_molecules.items():
        print("=" * 60)
        print(f"Test molecule: {mol_name}")
        print(f"SMILES: {smiles}")
        try:
            # Execute conversion
            mol_graph = smiles_to_graph(smiles, return_torch=True)
            # Validate and print conversion results
            print(f"✅ Conversion successful!")
            print(f"Number of atoms: {mol_graph['num_atoms']}")
            print(f"Number of bonds: {mol_graph['num_bonds']}")
            print(f"Atomic feature matrix shape: {mol_graph['x'].shape} | Expected: ({mol_graph['num_atoms']}, 39)")
            print(f"Edge index matrix shape: {mol_graph['edge_index'].shape} | Expected: (2, {2 * mol_graph['num_bonds']})")
            print(f"Bond feature matrix shape: {mol_graph['edge_attr'].shape} | Expected: ({2 * mol_graph['num_bonds']}, 10)")
            print(f"Atomic feature matrix: {mol_graph['x']}")
            print(f"Edge index matrix: {mol_graph['edge_index']}")
            print(f"Bond feature matrix: {mol_graph['edge_attr']}")
        except Exception as e:
            print(f"❌ Conversion failed: {str(e)}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()