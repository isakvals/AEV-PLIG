#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEV-PLIG Docking Rescoring Script

This script performs automated rescoring of docked ligand poses using the AEV-PLIG-based
graph neural network (GNN) model. It processes a given protein structure and a directory
of docked ligand files, converts them if necessary using openbabel, extracts molecular features,
generates molecular graphs, and predicts binding affinities.

Note: Automatic ligand conversion with OpenBabel can be inaccurate (e.g., incorrect bond orders).
For best results, manually convert ligands to SDF format and verify their correctness before rescoring.

Arguments:
    -p, --protein         Path to the input protein file (PDB or other formats convertible to PDB).
    -l, --ligands         Path to the directory containing ligand files (e.g., SDF, MOL, MOL2).
    --trained_model_name  Name of the trained model used for rescoring.
    --data_name           Identifier for the dataset (used for output file naming).
    --output_dir          Directory where output files (graphs and predictions) will be saved.
    -c, --num_workers     Number of parallel processes (default: all available cores).
    --device              Computation device ('auto', 'cpu', or a specific CUDA device index).
    --debug               Provide more logging/debug information while running the calculations.

Example Usage:
    python rescore_docking.py -p protein.pdb -l docked_ligands/ --num_workers 8

Author:
    Jochem Nelen (jnelen@ucam.edu)
"""

from typing import Any, List, Tuple, Optional, Dict
import argparse
import glob
import logging
import os
import pickle
import sys
import time

import qcelemental as qcel

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import GraphDatasetInference
from helpers import model_dict

from biopandas.pdb import PandasPdb
from openbabel import pybel
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit import RDLogger

from tqdm import tqdm

from torch_geometric.loader import DataLoader

# Suppress Torchani warnings
import warnings

warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings(
    "ignore", message="Dependency not satisfied, torchani.ase will not be available"
)
warnings.filterwarnings(
    "ignore", message="Dependency not satisfied, torchani.data will not be available"
)


import torch
import torchani
import torchani_mod


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Docking Rescoring Script: Process a protein and a directory of docked ligand files to generate AEV-PLIG rescoring predictions."
    )
    parser.add_argument(
        "-p", "--protein", type=str, required=True, help="Path to the protein file."
    )
    parser.add_argument(
        "-l",
        "--ligands",
        type=str,
        required=True,
        help="Directory containing ligand files (e.g., SDF or MOL) for docking rescoring.",
    )
    parser.add_argument(
        "--trained_model_name",
        type=str,
        default="model_GATv2Net_ligsim90_fep_benchmark",
        help="Name of the trained model for predictions (expected in output/trained_models).",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="docking_rescore",
        help="Dataset name to be used when saving graphs and predictions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory where outputs (graphs and predictions) will be saved.",
    )
    parser.add_argument(
        "-c",
        "--num_workers",
        "--cores",
        type=int,
        default=0,
        help="Number of workers for parallel processing. If set to 0, all available cores are used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for computation: 'auto' (select CUDA if available), 'cpu', or a specific CUDA device index (e.g., '0').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Provide more logging/debug information while running the calculations.",
    )
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--head", type=int, default=3)
    parser.add_argument("--activation_function", type=str, default="leaky_relu")

    return parser.parse_args()


def get_device(device_param: str) -> torch.device:
    """
    Determine and return the computation device.

    Args:
        device_param (str): Device parameter string ('auto', 'cpu', or CUDA index).

    Returns:
        torch.device: Selected computation device.
    """
    if device_param.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_param.lower() == "cpu":
        return torch.device("cpu")
    else:
        # Assume the user provided a valid CUDA device index
        if int(device_param) >= torch.cuda.device_count():
            raise ValueError(f"The CUDA device {device_param} doesn't seem to exist!")
        else:
            return torch.device(f"cuda:{device_param}")


def set_workers(num_workers: int) -> None:
    """
    Set the number of threads for parallel processing.

    Args:
        num_workers (int): Number of worker threads to use.
    """
    os.environ["OMP_NUM_THREADS"] = str(num_workers)
    os.environ["MKL_NUM_THREADS"] = str(num_workers)
    torch.set_num_threads(num_workers)


def validate_paths(config: argparse.Namespace) -> None:
    """
    Check if the provided protein file and ligand directory are valid.

    Args:
        config (argparse.Namespace): Configuration containing file paths.
    """
    if not os.path.isfile(config.protein):
        logging.error(f"Protein file not found: {config.protein}")
        sys.exit(1)

    if not os.path.isdir(config.ligands):
        logging.error(f"Ligand directory not found: {config.ligands}")
        sys.exit(1)

    logging.info(f"Using Protein file: {config.protein}")
    logging.info(f"Using Ligand directory: {config.ligands}")


def fix_formal_charge(mol: Chem.Mol) -> Chem.Mol:
    """
    Corrects formal charges for common elements in an RDKit molecule based on valence rules.

    This function adjusts the formal charge of atoms if their explicit valence deviates
    from expected chemical norms. It applies the following corrections:

    - Nitrogen (N): Should have valence 3 --> formal_charge = valence - 3
    - Oxygen (O): Should have valence 2 --> formal_charge = valence - 2
    - Sulfur (S): Should have valence <=6 --> formal_charge = valence - 6 (for hypervalent S)
    - Boron (B): Should have valence 3 --> formal_charge = 3 - valence

    Args:
        mol (Chem.Mol): RDKit molecule object with explicit valences.

    Returns:
        Chem.Mol: Molecule with corrected formal charges.
    """
    for atom in mol.GetAtoms():
        symbol: str = atom.GetSymbol()
        valence: int = atom.GetExplicitValence()

        if symbol == "N" and valence != 3:
            atom.SetFormalCharge(valence - 3)

        elif symbol == "O" and valence != 2:
            atom.SetFormalCharge(valence - 2)

        elif symbol == "S" and valence > 6:
            atom.SetFormalCharge(valence - 6)

        elif symbol == "B" and valence != 3:
            atom.SetFormalCharge(3 - valence)

    return mol


def mol_to_df(mol: Chem.Mol) -> pd.DataFrame:
    """
    Convert an RDKit molecule to a dataframe containing atom information.

    Args:
        mol (Chem.Mol): RDKit molecule object.

    Returns:
        pd.DataFrame: DataFrame with columns ATOM_INDEX, ATOM_TYPE, X, Y, Z.
    """
    atoms: List[List[Any]] = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    return df


def process_protein(protein_path: str, atom_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and process the protein file.

    Args:
        protein_path (str): Path to the protein file.
        atom_keys (pd.DataFrame): DataFrame mapping atom types.

    Returns:
        pd.DataFrame: Processed protein dataframe.
    """
    # Check file format
    file_extension = os.path.splitext(protein_path)[1].lower()

    if file_extension != ".pdb":
        # Try to convert the input file to PDB format using Open Babel
        try:
            protein_pybel = next(pybel.readfile(file_extension[1:], protein_path))
            pdb_text = protein_pybel.write("pdb")  # Get PDB format as a string
            logging.info("Automatically converted the input protein to PDB format.")
        except Exception as e:
            logging.error(f"Failed to convert the input protein to PDB format: {e}")
            raise ValueError(f"Conversion failed: {e}")

        # Try to read the converted PDB data as a DataFrame
        try:
            ppdb = PandasPdb()
            pdb_df = ppdb._construct_df(pdb_lines=pdb_text.splitlines(True))
        except Exception as e:
            logging.error(f"Failed to read the converted PDB file: {e}")
            raise ValueError("Failed to read the converted PDB file as a dataframe.")

    else:
        # Read the PDB file directly
        try:
            ppdb = PandasPdb().read_pdb(protein_path)
            pdb_df = ppdb.df
            pdb_text = ppdb.pdb_text
        except Exception as e:
            logging.error(f"Failed to read the PDB file: {e}")
            raise ValueError(f"Error reading PDB file: {e}")

    # Validate the PDB structure
    try:
        validate_protein(pdb_df["ATOM"], atom_keys)
    except Exception as e:
        logging.error(f"Failed to validate the PDB file: {e}")
        raise ValueError(f"Error validating the PDB file: {e}")

    logging.info("Validated the protein structure")

    # Prepare final protein_df (LoadPDBasDF_old method from process_and_predict.py)
    prot_atoms: List[List[Any]] = []
    for i in pdb_text.splitlines(True):
        if i[:4] == "ATOM":
            # Include only non-hydrogen atoms
            if (
                len(i[12:16].replace(" ", "")) < 4
                and i[12:16].replace(" ", "")[0] != "H"
            ) or (
                len(i[12:16].replace(" ", "")) == 4
                and i[12:16].replace(" ", "")[1] != "H"
                and i[12:16].replace(" ", "")[0] != "H"
            ):
                prot_atoms.append(
                    [
                        int(i[6:11]),
                        i[17:20] + "-" + i[12:16].replace(" ", ""),
                        float(i[30:38]),
                        float(i[38:46]),
                        float(i[46:54]),
                    ]
                )

    pdb_df_final = pd.DataFrame(
        prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"]
    )
    pdb_df_final = (
        pdb_df_final.merge(atom_keys, left_on="PDB_ATOM", right_on="PDB_ATOM")[
            ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
        ]
        .sort_values(by="ATOM_INDEX")
        .reset_index(drop=True)
    )

    return pdb_df_final


def process_ligand(
    ligand_path: str,
    processed_protein: pd.DataFrame,
    atom_map: pd.DataFrame,
    radial_coefs: List[Any],
) -> Tuple[str, Optional[Any], Optional[str]]:
    """
    Processes a single ligand file: validates format, converts if necessary using openbabel,
    checks for unsupported elements/bonds, and constructs a molecular graph.

    Args:
        ligand_path (str): Path to the ligand file.
        processed_protein (pd.DataFrame): Processed protein DataFrame.
        atom_map (pd.DataFrame): Atom mapping for AEV computation.
        radial_coefs (List[Any]): Radial coefficients [RcR, EtaR, RsR].

    Returns:
        tuple: (ligand_name, graph, None) if successful,
               (ligand_name, None, error_message) if an error occurs.
    """
    file_name, file_extension = os.path.splitext(os.path.basename(ligand_path))
    file_extension = file_extension.lower()

    lig = None

    # Try loading the molecule: either directly via RDKit (SDF or mol2) or convert using Open Babel for other formats (pdb(qt))
    if file_extension == ".sdf":
        suppl = Chem.SDMolSupplier(ligand_path, removeHs=False)
        lig = suppl[0]
    elif file_extension == ".mol2":
        lig = Chem.MolFromMol2File(ligand_path, removeHs=False)
    else:
        try:
            ligand_pybel = next(pybel.readfile(file_extension[1:], ligand_path))
        except Exception:
            return file_name, None, "OpenBabel conversion error"
        try:
            lig = Chem.MolFromMolBlock(
                ligand_pybel.write("sdf"), removeHs=False, sanitize=False
            )
            fix_formal_charge(lig)
            Chem.SanitizeMol(lig)
        except Exception:
            return file_name, None, "RDKit sanitization error"

    # Check if RDKit successfully read the molecule
    if lig is None:
        return file_name, None, "RDKit read error"

    # Check for unsupported elements
    allowed_elements = {"F", "N", "Cl", "O", "Br", "C", "B", "P", "I", "S"}
    mol_df = mol_to_df(lig)

    if not set(mol_df["ATOM_TYPE"].values).issubset(allowed_elements):
        return file_name, None, "Contains unsupported elements"

    # Check for unspecified bonds
    for bond in lig.GetBonds():
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            return file_name, None, "Contains unspecified bonds"

    # Try converting to graph
    try:
        aevs = GetMolAEVs_extended(processed_protein, mol_df, radial_coefs, atom_map)
        graph = mol_to_graph(lig, mol_df, aevs)
        return file_name, graph, None
    except Exception as e:
        return file_name, None, f"Graph conversion error: {str(e)}"


def validate_protein(protein: pd.DataFrame, atom_keys: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the protein DataFrame based on allowed residues and atom types.

    Args:
        protein (pd.DataFrame): Protein DataFrame extracted from a PDB file.
        atom_keys (pd.DataFrame): DataFrame containing allowed atom type mappings.

    Returns:
        pd.DataFrame: Cleaned protein DataFrame.
    """
    allowed_residues = atom_keys["RESIDUE"].unique()

    protein = protein[~protein["atom_name"].str.startswith("H")]
    protein = protein[~protein["atom_name"].str.startswith(tuple(map(str, range(10))))]

    discarded_residues = protein[~protein["residue_name"].isin(allowed_residues)]
    if len(discarded_residues) > 0:
        logging.warning("WARNING: Protein contains unsupported residues.")
        logging.warning("Ignoring following residues:")
        logging.warning(discarded_residues["residue_name"].unique())

    protein = protein[protein["residue_name"].isin(allowed_residues)]
    protein["PDB_ATOM"] = protein["residue_name"] + "-" + protein["atom_name"]
    protein = protein[
        ["atom_number", "PDB_ATOM", "x_coord", "y_coord", "z_coord"]
    ].rename(
        columns={
            "atom_number": "ATOM_INDEX",
            "x_coord": "X",
            "y_coord": "Y",
            "z_coord": "Z",
        }
    )
    protein = (
        protein.merge(atom_keys, how="left", on="PDB_ATOM")
        .sort_values(by="ATOM_INDEX")
        .reset_index(drop=True)
    )

    if list(protein["ATOM_TYPE"].isna()).count(True) > 0:
        logging.warning("WARNING: Protein contains unsupported atom types.")
        logging.warning("Ignoring following atom types:")
        logging.warning(protein[protein["ATOM_TYPE"].isna()]["PDB_ATOM"].unique())
    return protein


def atom_features(
    atom: Chem.rdchem.Atom,
    features: List[str] = [
        "atom_symbol",
        "num_heavy_atoms",
        "total_num_Hs",
        "explicit_valence",
        "is_aromatic",
        "is_in_ring",
    ],
) -> np.ndarray:
    """
    Computes the ligand atom features for graph node construction.

    The standard features are:
        - atom_symbol: one hot encoding of atom symbol
        - num_heavy_atoms: number of heavy atom neighbors
        - total_num_Hs: number of hydrogen atom neighbors
        - explicit_valence: explicit valence of the atom
        - is_aromatic: boolean (1 if aromatic, 0 otherwise)
        - is_in_ring: boolean (1 if in a ring, 0 otherwise)

    Args:
        atom (Chem.rdchem.Atom): An RDKit atom.
        features (List[str], optional): List of features to compute.

    Returns:
        np.ndarray: Array of computed features.
    """
    feature_list: List[Any] = []
    if "atom_symbol" in features:
        feature_list.extend(
            one_of_k_encoding(
                atom.GetSymbol(), ["F", "N", "Cl", "O", "Br", "C", "B", "P", "I", "S"]
            )
        )
    if "num_heavy_atoms" in features:
        feature_list.append(
            len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"])
        )
    if "total_num_Hs" in features:
        feature_list.append(
            len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"])
        )
    if "explicit_valence" in features:
        feature_list.append(atom.GetExplicitValence())
    if "is_aromatic" in features:
        feature_list.append(1 if atom.GetIsAromatic() else 0)
    if "is_in_ring" in features:
        feature_list.append(1 if atom.IsInRing() else 0)

    return np.array(feature_list)


def GetMolAEVs_extended(
    processed_protein: pd.DataFrame,
    mol_df: pd.DataFrame,
    radial_coefs: List[Any],
    atom_map: pd.DataFrame,
) -> torch.Tensor:
    """
    Computes Atomic Environment Vectors (AEVs) for a given protein-ligand complex.

    Args:
        processed_protein (pd.DataFrame): Processed protein DataFrame.
        mol_df (pd.DataFrame): DataFrame of ligand atoms.
        radial_coefs (List[Any]): Radial coefficients [RcR, EtaR, RsR].
        atom_map (pd.DataFrame): Atom mapping for AEV computation.

    Returns:
        torch.Tensor: Computed AEV tensor for the ligand atoms.
    """
    with torch.no_grad():
        # Load protein and ligand as DataFrames
        Target = processed_protein
        Ligand = mol_df

        # Extract radial coefficients
        RcR, EtaR, RsR = radial_coefs

        # Angular coefficients (GA)
        RcA, Zeta, TsA, EtaA, RsA = (
            2.0,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([1.0]),
        )

        # **Vectorized filtering of Target based on RcR cutoff**
        distance_cutoff = RcR + 0.1
        min_bounds = Ligand[["X", "Y", "Z"]].min().values - distance_cutoff
        max_bounds = Ligand[["X", "Y", "Z"]].max().values + distance_cutoff
        Target = Target[
            (Target[["X", "Y", "Z"]] >= min_bounds).all(axis=1)
            & (Target[["X", "Y", "Z"]] <= max_bounds).all(axis=1)
        ]

        # Merge atom types for AEV calculation
        Target = Target.merge(atom_map, on="ATOM_TYPE", how="left")

        # **Create tensors for atomic numbers and coordinates**
        mol_len = len(Ligand)

        # Instead of np.append(), directly create a torch tensor
        atomicnums = torch.cat(
            [
                torch.full(
                    (mol_len,), 6, dtype=torch.int64
                ),  # Treat molecule atoms as Carbon (C=6)
                torch.tensor(Target["ATOM_NR"].values, dtype=torch.int64),
            ]
        ).unsqueeze(0)

        # Directly concatenate coordinates from both ligand and protein
        coordinates = torch.tensor(
            pd.concat([Ligand[["X", "Y", "Z"]], Target[["X", "Y", "Z"]]]).values,
            dtype=torch.float32,
        ).unsqueeze(0)

        # **Precompute atom symbols list (faster list comprehension)**
        atom_symbols = [qcel.periodictable.to_symbol(i) for i in range(1, 23)]

        # **Initialize AEV Computer**
        AEVC = torchani_mod.AEVComputer(
            RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, len(atom_symbols)
        )

        # **Compute AEVs**
        SC = torchani.SpeciesConverter(atom_symbols)
        sc = SC((atomicnums, coordinates))
        aev = AEVC.forward((sc.species, sc.coordinates), torch.tensor(mol_len))

        # **Extract only radial terms**
        n = len(atom_symbols)
        n_rad_sub = len(EtaR) * len(RsR)
        indices = list(range(n * n_rad_sub))

        return aev.aevs.squeeze(0)[:mol_len, indices]


def mol_to_graph(
    mol: Chem.Mol,
    mol_df: pd.DataFrame,
    aevs: torch.Tensor,
    extra_features: List[str] = [
        "atom_symbol",
        "num_heavy_atoms",
        "total_num_Hs",
        "explicit_valence",
        "is_aromatic",
        "is_in_ring",
    ],
) -> Tuple[int, List[np.ndarray], List[List[int]], List[List[float]]]:
    """
    Converts an RDKit molecule to a graph representation.

    Args:
        mol (Chem.Mol): RDKit molecule.
        mol_df (pd.DataFrame): DataFrame of molecule atoms.
        aevs (torch.Tensor): Precomputed Atomic Environment Vectors.
        extra_features (List[str], optional): List of extra features for nodes.

    Returns:
        tuple: (number of nodes, list of node features, edge_index, edge_attr)
    """
    features: List[np.ndarray] = []
    heavy_atom_index: List[int] = []
    idx_to_idx: Dict[int, int] = {}
    counter = 0

    # Generate nodes
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df["ATOM_INDEX"] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            feature = np.append(atom_features(atom), aevs[aev_idx, :])
            features.append(feature)
            counter += 1

    # Generate edges
    edges: List[List[Any]] = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            bond_type = one_of_k_encoding(bond.GetBondType(), [1, 12, 2, 3])
            bond_type = [float(b) for b in bond_type]
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)

    df = pd.DataFrame(
        edges, columns=["atom1", "atom2", "single", "aromatic", "double", "triple"]
    )
    df = df.sort_values(by=["atom1", "atom2"])

    edge_index = df[["atom1", "atom2"]].to_numpy().tolist()
    edge_attr = df[["single", "aromatic", "double", "triple"]].to_numpy().tolist()

    return len(mol_df), features, edge_index, edge_attr


def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
    """
    One-hot encodes a value based on the allowable set.

    Args:
        x (Any): Value to encode.
        allowable_set (List[Any]): List of allowable values.

    Returns:
        List[bool]: One-hot encoded list.
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def perform_inference(
    config: argparse.Namespace, graphs_dict: Dict[str, Any]
) -> pd.DataFrame:
    """
    Runs inference using a trained model on preprocessed ligand graphs.

    Args:
        config (argparse.Namespace): Configuration parameters.
        graphs_dict (Dict[str, Any]): Dictionary of precomputed ligand graphs.

    Returns:
        pd.DataFrame: DataFrame containing predictions merged with test IDs.
    """
    # Load trained model scaler (if needed)
    model_name = config.trained_model_name
    scaler_path = f"output/trained_models/{model_name}.pickle"

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Prepare dataset
    test_ids = list(graphs_dict.keys())
    test_graph_ids = list(range(len(test_ids)))  # Unique graph indices

    test_data = GraphDatasetInference(
        ids=test_ids, graph_ids=test_graph_ids, graphs_dict=graphs_dict
    )

    # Use efficient batch size
    batch_size = min(512, len(test_data))  # Auto-adjust batch size based on data size
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model
    modeling = model_dict["GATv2Net"]
    model = modeling(
        node_feature_dim=test_data.num_node_features,
        edge_feature_dim=test_data.num_edge_features,
        config=config,
    )

    # Run inference across multiple models and ensemble results
    with torch.inference_mode():
        for i in tqdm(range(10), desc="Performing Inference"):
            model_path = f"output/trained_models/{config.trained_model_name}_{i}.model"
            model.load_state_dict(torch.load(model_path, map_location=config.device))

            graph_ids_test, P_test = predict(model, config.device, test_loader, scaler)

            if i == 0:
                df_test = pd.DataFrame({"graph_id": graph_ids_test})

            df_test[f"preds_{i}"] = P_test

    # Compute mean predictions
    df_test["preds"] = df_test.iloc[:, 1:].mean(axis=1)

    # Merge predictions with test IDs
    results_df = pd.DataFrame({"unique_id": test_ids}).merge(
        df_test.drop(columns=["graph_id"]), left_index=True, right_index=True
    )
    results_df = results_df.sort_values(by="preds", ascending=False)

    return results_df


def predict(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    y_scaler: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using a given model and data loader.

    Parameters:
        model (torch.nn.Module): The model to use for predictions.
        device (torch.device): The device (CPU or GPU) to perform computations.
        loader (DataLoader): DataLoader containing the dataset.
        y_scaler (Optional[Any]): Scaler to inverse transform predictions, if provided.

    Returns:
        tuple: (graph_ids, predictions)
            - graph_ids: Array of identifiers.
            - predictions: Array of model predictions (inverse-transformed if y_scaler provided).
    """
    model.eval()
    model.to(device)

    preds_list: List[torch.Tensor] = []
    graph_ids_list: List[torch.Tensor] = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            preds_list.append(output)

            graph_ids_list.append(data.y.view(-1, 1))

    # Concatenate all outputs at once
    total_preds = torch.cat(preds_list, dim=0)
    total_graph_ids = torch.cat(graph_ids_list, dim=0)

    # Move to CPU and convert to numpy arrays
    total_graph_ids = total_graph_ids.cpu().numpy().flatten()
    total_preds_np = total_preds.cpu().numpy().reshape(-1, 1)

    if y_scaler is not None:
        total_preds_np = y_scaler.inverse_transform(total_preds_np).flatten()
    else:
        total_preds_np = total_preds_np.flatten()

    return total_graph_ids, total_preds_np


if __name__ == "__main__":
    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_time = time.time()
    config = parse_args()

    if not config.debug:
        # Suppress openbabel messaging
        ob.obErrorLog.SetOutputLevel(-1)
        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)
        logging.getLogger("torch_geometric").setLevel(logging.WARNING)

    # Determine the number of workers
    if config.num_workers <= 0:
        config.num_workers = os.cpu_count()
        logging.info(f"Using all available cores: {config.num_workers} workers.")
    else:
        logging.info(f"Using {config.num_workers} worker(s).")

    # Set the workers to 1 so CPUs don't get overloaded while running in parallel using ProcessPoolExecutor
    set_workers(1)

    config.device = get_device(config.device)
    logging.info(f"Using {config.device} for inference")

    # Validate the input protein and ligand paths
    validate_paths(config)

    # Load atom_keys
    atom_keys = pd.read_csv("data/PDB_Atom_Keys.csv", sep=",")
    atom_keys["RESIDUE"] = atom_keys["PDB_ATOM"].apply(lambda x: x.split("-")[0])

    # Process the protein structure
    processed_protein = process_protein(config.protein, atom_keys)

    logging.info(
        f"Finished preprocessing of the protein in {time.time()-start_time:.2f}s"
    )

    # Get a list of all the ligand files using glob
    ligand_paths = glob.glob(f"{config.ligands}/*.*")

    logging.info(f"Identified {len(ligand_paths)} files in {config.ligands}")

    processed_ligands: Dict[str, Any] = {}
    failed_ligands: List[Tuple[str, str]] = []

    # Create output directory if it doesn't exist yet
    os.makedirs(config.output_dir, exist_ok=True)

    # Precompute some metrics to make graph generation more efficient

    # Radial coefficients: ANI-2x
    RcR = 5.1  # Radial cutoff
    EtaR = torch.tensor([19.7])  # Radial decay
    RsR = torch.tensor(
        [
            0.80,
            1.07,
            1.34,
            1.61,
            1.88,
            2.14,
            2.41,
            2.68,
            2.95,
            3.22,
            3.49,
            3.76,
            4.03,
            4.29,
            4.56,
            4.83,
        ]
    )  # Radial shift
    radial_coefs = [RcR, EtaR, RsR]

    # Reload atom_keys
    atom_keys = pd.read_csv("data/PDB_Atom_Keys.csv", sep=",")

    # Define atom_map
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    process_ligand_partial = partial(
        process_ligand,
        processed_protein=processed_protein,
        atom_map=atom_map,
        radial_coefs=radial_coefs,
    )

    # Validate and process ligands into graphs
    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        futures = {
            executor.submit(process_ligand_partial, ligand_path): ligand_path
            for ligand_path in ligand_paths
        }

        # Use tqdm for progress tracking
        for future in tqdm(
            as_completed(futures), desc="Generating Graphs", total=len(futures)
        ):
            ligand_path = futures[future]  # Get ligand path from the future dict

            try:
                file_name, graph, error = future.result()

                if error:
                    failed_ligands.append((file_name, error))
                else:
                    processed_ligands[file_name] = graph

            except Exception as e:
                failed_ligands.append((ligand_path, str(e)))  # Catch unexpected errors
                logging.error(
                    f"Unexpected error processing {ligand_path}: {e}", exc_info=True
                )

    logging.info(
        f"Successfully converted {len(processed_ligands)} ligands to graphs in {time.time()-start_time:.2f}s"
    )

    # Report any failed ligands
    if failed_ligands:
        logging.warning(f"{len(failed_ligands)} ligand(s) failed to be processed.")

        # Format failed ligands output
        failure_summary = "\n".join(
            [f"  - {file_name}: {error}" for file_name, error in failed_ligands]
        )

        with open(
            f"{config.output_dir}/{config.data_name}_failed_ligands.log", "w"
        ) as f:
            f.write(f"Failed ligands ({len(failed_ligands)} total):\n")
            f.write(failure_summary + "\n")

        logging.info("Failure report saved to 'failed_ligands.log'.")

    # Set the workers again to the maximal amount to ensure optimal inference
    set_workers(config.num_workers)

    # Start inference
    inference_start_time = time.time()

    results_df = perform_inference(config, processed_ligands)

    # Save predictions
    output_file = f"{config.output_dir}/{config.data_name}_predictions.csv"

    logging.info(f"Saving predictions to {output_file}")
    results_df.to_csv(output_file, index=False, float_format="%.3f")

    logging.info(
        f"Finished inference for {len(results_df)} compounds in {time.time()-inference_start_time:.2f}s"
    )
    logging.info(f"Total run time: {time.time()-start_time:.2f}s")
