#!/usr/bin/env python
"""
Molecular Similarity Filter Script
==================================

This script processes an SDF file containing chemical compounds and filters them based
on structural similarity using Morgan fingerprints and Tanimoto coefficient (Tc).

Purpose:
--------
The script identifies and extracts compounds that are structurally unique within the dataset
according to two different similarity thresholds:
1. Compounds that don't have any neighbors with Tc > 0.7 (moderately unique)
2. Compounds that don't have any neighbors with Tc > 0.5 (highly unique)

Technical details:
-----------------
- Molecules are first converted to canonical SMILES to ensure consistent representation
- Morgan fingerprints (ECFP4, radius=2, 2048 bits) are calculated for each compound
- A full pairwise similarity matrix is built using Tanimoto coefficient
- Compounds are filtered based on their maximum similarity to any other compound

Usage:
------
python remove-similars-from-sdf.py <sdf_file>

Output:
-------
Two SDF files are created:
- <input_name>_tc07.sdf: Contains compounds with no neighbors above Tc > 0.7
- <input_name>_tc05.sdf: Contains compounds with no neighbors above Tc > 0.5

Requirements:
------------
- RDKit
- NumPy

Author: Serhii Vakal, Orion Pharma, Turku, Finland
Date: May 2025
"""

import os
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs


def process_sdf_file(sdf_file):
    # Check if the file exists
    if not os.path.isfile(sdf_file):
        print(f"Error: File '{sdf_file}' does not exist.")
        return False
    
    # Get the base name of the file (without extension) for output files
    base_name = os.path.splitext(os.path.basename(sdf_file))[0]
    output_file_07 = f"{base_name}_tc07.sdf"
    output_file_05 = f"{base_name}_tc05.sdf"
    
    print(f"Reading molecules from {sdf_file}...")
    
    # Read molecules from SDF file
    mols = []
    supplier = Chem.SDMolSupplier(sdf_file)
    for mol in supplier:
        if mol is not None:
            mols.append(mol)
    
    if not mols:
        print("No valid molecules found in the input file.")
        return False
    
    print(f"Found {len(mols)} valid molecules. Processing...")
    
    # Generate canonical SMILES and Morgan fingerprints
    canonical_mols = []
    fps = []
    
    for mol in mols:
        # Generate canonical SMILES
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        # Create a new molecule from the canonical SMILES
        canonical_mol = Chem.MolFromSmiles(smiles)
        
        if canonical_mol is None:
            print(f"Warning: Could not generate canonical molecule for {Chem.MolToSmiles(mol)}. Skipping.")
            continue
        
        # Copy properties from original molecule
        for prop_name in mol.GetPropNames():
            canonical_mol.SetProp(prop_name, mol.GetProp(prop_name))
        
        canonical_mols.append(canonical_mol)
        
        # Calculate Morgan fingerprint (ECFP4)
        fp = AllChem.GetMorganFingerprintAsBitVect(canonical_mol, 2, nBits=2048)
        fps.append(fp)
    
    # Calculate Tanimoto similarity matrix
    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            tc = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarity_matrix[i, j] = tc
            similarity_matrix[j, i] = tc  # Matrix is symmetric
    
    # Find molecules that don't have any neighbors with Tc > 0.7
    tc_07_indices = []
    for i in range(n):
        # Check all other molecules (skip self-comparison where Tc = 1.0)
        if not any(j != i and similarity_matrix[i, j] > 0.7 for j in range(n)):
            tc_07_indices.append(i)
    
    # Find molecules that don't have any neighbors with Tc > 0.5
    tc_05_indices = []
    for i in range(n):
        # Check all other molecules (skip self-comparison where Tc = 1.0)
        if not any(j != i and similarity_matrix[i, j] > 0.5 for j in range(n)):
            tc_05_indices.append(i)
    
    # Write output files
    if tc_07_indices:
        print(f"Writing {len(tc_07_indices)} molecules to {output_file_07}...")
        writer = Chem.SDWriter(output_file_07)
        for idx in tc_07_indices:
            writer.write(canonical_mols[idx])
        writer.close()
    else:
        print(f"No molecules found for Tc > 0.7 filter")
    
    if tc_05_indices:
        print(f"Writing {len(tc_05_indices)} molecules to {output_file_05}...")
        writer = Chem.SDWriter(output_file_05)
        for idx in tc_05_indices:
            writer.write(canonical_mols[idx])
        writer.close()
    else:
        print(f"No molecules found for Tc > 0.5 filter")
    
    print("Processing complete!")
    return True


if __name__ == "__main__":
    # Check if SDF file argument is provided
    if len(sys.argv) != 2:
        print("Usage: python filter_by_similarity.py <sdf_file>")
        sys.exit(1)
    
    # Get SDF file from command-line argument
    sdf_file = sys.argv[1]
    
    # Process the SDF file
    if not process_sdf_file(sdf_file):
        sys.exit(1)