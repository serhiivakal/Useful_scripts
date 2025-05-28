#!/usr/bin/env python3
"""
Aromatic Planarity Analyzer (aromatic_planarity_analyzer.py)
===========================================================

A tool for analyzing the planarity of aromatic rings in 3D molecular structures.

Description:
-----------
This script evaluates the planarity of aromatic rings in molecular structures by 
calculating the deviation of ring atoms from their best-fit plane. It applies a 
tiered classification system to rate both individual rings and entire molecules (poses)
on their planarity quality.

Classification Thresholds:
-------------------------
- Excellent:   < 0.02Å RMSD from plane
- Acceptable:  0.02-0.05Å RMSD from plane
- Borderline:  0.05-0.08Å RMSD from plane
- Violation:   > 0.08Å RMSD from plane

How It Works:
-----------
1. The script reads 3D molecular structures from an SDF file
2. For each molecule, it identifies all aromatic rings
3. For each ring, it calculates the best-fit plane and the RMSD of atoms from this plane
4. It classifies each ring and overall molecule based on the calculated deviations
5. It compiles statistics on both ring-level and pose-level planarity
6. It reports detailed results and can export problematic structures

Usage:
-----
Basic usage:
    python aromatic_planarity_analyzer.py molecules.sdf

Save results to CSV:
    python aromatic_planarity_analyzer.py molecules.sdf -o results.csv

Export molecules with violations:
    python aromatic_planarity_analyzer.py molecules.sdf --write-violations bad_poses.sdf

Export molecules for all categories:
    python aromatic_planarity_analyzer.py molecules.sdf --write-all-categories --output-prefix quality_

Options:
-------
  -o, --output           Output CSV file for results
  -q, --quiet            Suppress detailed output
  --write-violations     Write molecules with violations (>0.08Å) to SDF
  --write-borderline     Write molecules with borderline cases (0.05-0.08Å) to SDF
  --write-all-categories Write separate SDF files for each planarity category
  --output-prefix        Prefix for SDF output files when using --write-all-categories

Requirements:
------------
- RDKit
- NumPy
- pandas

Developed by Dr. Serhii Vakal, Orion Pharma, Turku, 2025.
Available under MIT license.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Descriptors
import argparse
import sys
from typing import List, Tuple, Dict, Optional
import pandas as pd
from pathlib import Path


# Define planarity thresholds
PLANARITY_THRESHOLDS = {
    'excellent': 0.02,   # < 0.02Å: Excellent planarity
    'acceptable': 0.05,  # 0.02-0.05Å: Acceptable planarity
    'borderline': 0.08,  # 0.05-0.08Å: Borderline cases requiring inspection
    'violation': float('inf')  # > 0.08Å: Non-planar, problematic
}


def calculate_plane_deviation(coords: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calculate RMSD deviation from best-fit plane for a set of coordinates.
    
    Args:
        coords: Nx3 array of atomic coordinates
        
    Returns:
        rmsd: Root mean square deviation from plane
        normal: Normal vector of the best-fit plane
    """
    # Center the coordinates
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    
    # SVD to find the best-fit plane
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2]  # Normal vector is the last row
    
    # Calculate distances from plane
    distances = np.abs(np.dot(centered, normal))
    rmsd = np.sqrt(np.mean(distances**2))
    
    return rmsd, normal


def get_aromatic_rings(mol: Chem.Mol) -> List[List[int]]:
    """
    Get all aromatic rings in the molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        List of atom indices for each aromatic ring
    """
    ri = mol.GetRingInfo()
    aromatic_rings = []
    
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatic_rings.append(list(ring))
    
    return aromatic_rings


def classify_planarity(rmsd: float) -> str:
    """
    Classify planarity based on RMSD deviation.
    
    Args:
        rmsd: RMSD deviation from planarity in Angstroms
        
    Returns:
        Planarity classification as a string
    """
    if rmsd < PLANARITY_THRESHOLDS['excellent']:
        return 'excellent'
    elif rmsd < PLANARITY_THRESHOLDS['acceptable']:
        return 'acceptable'
    elif rmsd < PLANARITY_THRESHOLDS['borderline']:
        return 'borderline'
    else:
        return 'violation'


def analyze_ring_planarity(mol: Chem.Mol, ring_atoms: List[int]) -> Dict:
    """
    Analyze planarity of a single ring.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        ring_atoms: List of atom indices in the ring
        
    Returns:
        Dictionary with planarity metrics
    """
    conf = mol.GetConformer()
    
    # Get coordinates for ring atoms
    coords = np.array([conf.GetAtomPosition(idx) for idx in ring_atoms])
    
    # Calculate planarity deviation
    rmsd, normal = calculate_plane_deviation(coords)
    
    # Calculate max deviation
    centered = coords - np.mean(coords, axis=0)
    distances = np.abs(np.dot(centered, normal))
    max_deviation = np.max(distances)
    
    # Get ring size and atom types
    atom_types = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring_atoms]
    
    # Classify planarity
    planarity_class = classify_planarity(rmsd)
    
    return {
        'ring_size': len(ring_atoms),
        'atom_indices': ring_atoms,
        'atom_types': atom_types,
        'rmsd_deviation': rmsd,
        'max_deviation': max_deviation,
        'planarity_class': planarity_class
    }


def analyze_molecule(mol: Chem.Mol, mol_name: str = None) -> Optional[Dict]:
    """
    Analyze all aromatic rings in a molecule for planarity.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        mol_name: Optional molecule name
        
    Returns:
        Dictionary with analysis results or None if analysis fails
    """
    if mol is None:
        return None
        
    # Ensure we have 3D coordinates
    if mol.GetNumConformers() == 0:
        return None
    
    aromatic_rings = get_aromatic_rings(mol)
    
    if not aromatic_rings:
        return {
            'name': mol_name or 'Unknown',
            'num_aromatic_rings': 0,
            'rings': [],
            'pose_planarity': 'no_rings'  # No rings to evaluate
        }
    
    ring_analyses = []
    worst_planarity = 'excellent'  # Start with best category
    
    for i, ring in enumerate(aromatic_rings):
        ring_data = analyze_ring_planarity(mol, ring)
        ring_data['ring_id'] = i
        ring_analyses.append(ring_data)
        
        # Track worst planarity class for the molecule
        if ring_data['planarity_class'] == 'violation':
            worst_planarity = 'violation'
        elif ring_data['planarity_class'] == 'borderline' and worst_planarity != 'violation':
            worst_planarity = 'borderline'
        elif ring_data['planarity_class'] == 'acceptable' and worst_planarity not in ['violation', 'borderline']:
            worst_planarity = 'acceptable'
    
    # Calculate summary statistics
    all_rmsds = [r['rmsd_deviation'] for r in ring_analyses]
    planarity_counts = {
        'excellent': sum(1 for r in ring_analyses if r['planarity_class'] == 'excellent'),
        'acceptable': sum(1 for r in ring_analyses if r['planarity_class'] == 'acceptable'),
        'borderline': sum(1 for r in ring_analyses if r['planarity_class'] == 'borderline'),
        'violation': sum(1 for r in ring_analyses if r['planarity_class'] == 'violation')
    }
    
    return {
        'name': mol_name or 'Unknown',
        'num_aromatic_rings': len(aromatic_rings),
        'planarity_counts': planarity_counts,
        'max_rmsd': max(all_rmsds) if all_rmsds else 0,
        'mean_rmsd': np.mean(all_rmsds) if all_rmsds else 0,
        'pose_planarity': worst_planarity,  # Overall pose classification based on worst ring
        'rings': ring_analyses
    }


def process_sdf_file(sdf_path: str, output_csv: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Process an SDF file and analyze all molecules for aromatic planarity.
    
    Args:
        sdf_path: Path to SDF file
        output_csv: Optional path to save results as CSV
        verbose: Print detailed results
        
    Returns:
        DataFrame with analysis results
    """
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    
    results = []
    mol_results = []
    
    # Ring-level statistics
    total_rings = 0
    planarity_totals = {
        'excellent': 0,
        'acceptable': 0,
        'borderline': 0,
        'violation': 0
    }
    
    # Pose-level statistics
    pose_totals = {
        'no_rings': 0,
        'excellent': 0,
        'acceptable': 0,
        'borderline': 0,
        'violation': 0
    }
    
    for idx, mol in enumerate(supplier):
        if mol is None:
            if verbose:
                print(f"Warning: Could not read molecule {idx}")
            continue
            
        # Get molecule name
        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else f'Mol_{idx}'
        
        # Analyze molecule
        analysis = analyze_molecule(mol, mol_name)
        if analysis:
            mol_results.append(analysis)
            
            # Update ring-level statistics
            if 'planarity_counts' in analysis:
                for category, count in analysis['planarity_counts'].items():
                    planarity_totals[category] += count
                total_rings += analysis['num_aromatic_rings']
            
            # Update pose-level statistics
            pose_totals[analysis['pose_planarity']] += 1
                
            # Flatten results for DataFrame
            for ring in analysis.get('rings', []):
                results.append({
                    'molecule': mol_name,
                    'pose_planarity': analysis['pose_planarity'],
                    'ring_id': ring['ring_id'],
                    'ring_size': ring['ring_size'],
                    'atom_indices': str(ring['atom_indices']),
                    'atom_types': ''.join(ring['atom_types']),
                    'rmsd_deviation': ring['rmsd_deviation'],
                    'max_deviation': ring['max_deviation'],
                    'planarity_class': ring['planarity_class']
                })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate total poses (molecules) with at least one aromatic ring
    total_poses_with_rings = sum(pose_totals.values()) - pose_totals['no_rings']
    total_poses = sum(pose_totals.values())
    
    # Print summary
    if verbose:
        print("\n=== AROMATIC PLANARITY ANALYSIS SUMMARY ===")
        print(f"Total molecules analyzed: {total_poses}")
        print(f"Molecules with aromatic rings: {total_poses_with_rings}")
        print(f"Molecules without aromatic rings: {pose_totals['no_rings']}")
        print(f"Total aromatic rings: {total_rings}")
        
        print("\n=== RING-LEVEL PLANARITY STATISTICS ===")
        if total_rings > 0:
            print(f"  Excellent  (<{PLANARITY_THRESHOLDS['excellent']}Å): {planarity_totals['excellent']} rings " +
                  f"({planarity_totals['excellent']/total_rings*100:.1f}%)")
            print(f"  Acceptable ({PLANARITY_THRESHOLDS['excellent']}-{PLANARITY_THRESHOLDS['acceptable']}Å): {planarity_totals['acceptable']} rings " +
                  f"({planarity_totals['acceptable']/total_rings*100:.1f}%)")
            print(f"  Borderline ({PLANARITY_THRESHOLDS['acceptable']}-{PLANARITY_THRESHOLDS['borderline']}Å): {planarity_totals['borderline']} rings " +
                  f"({planarity_totals['borderline']/total_rings*100:.1f}%)")
            print(f"  Violation  (>{PLANARITY_THRESHOLDS['borderline']}Å): {planarity_totals['violation']} rings " +
                  f"({planarity_totals['violation']/total_rings*100:.1f}%)")
        else:
            print("  No aromatic rings found in the dataset")
        
        print("\n=== POSE-LEVEL PLANARITY STATISTICS ===")
        if total_poses_with_rings > 0:
            print(f"  Excellent  (all rings <{PLANARITY_THRESHOLDS['excellent']}Å): {pose_totals['excellent']} poses " +
                  f"({pose_totals['excellent']/total_poses_with_rings*100:.1f}%)")
            print(f"  Acceptable (worst ring {PLANARITY_THRESHOLDS['excellent']}-{PLANARITY_THRESHOLDS['acceptable']}Å): {pose_totals['acceptable']} poses " +
                  f"({pose_totals['acceptable']/total_poses_with_rings*100:.1f}%)")
            print(f"  Borderline (worst ring {PLANARITY_THRESHOLDS['acceptable']}-{PLANARITY_THRESHOLDS['borderline']}Å): {pose_totals['borderline']} poses " +
                  f"({pose_totals['borderline']/total_poses_with_rings*100:.1f}%)")
            print(f"  Violation  (any ring >{PLANARITY_THRESHOLDS['borderline']}Å): {pose_totals['violation']} poses " +
                  f"({pose_totals['violation']/total_poses_with_rings*100:.1f}%)")
        else:
            print("  No poses with aromatic rings found in the dataset")
        
        if len(df) > 0:
            print(f"\nWorst violations:")
            print(df.nlargest(5, 'rmsd_deviation')[['molecule', 'ring_id', 'atom_types', 'rmsd_deviation', 'planarity_class']])
            
            # Per-molecule summary
            print("\n=== PER-MOLECULE SUMMARY ===")
            if not df.empty:
                # Group by molecule and get the worst planarity class and statistics
                mol_summary = df.groupby('molecule').agg({
                    'pose_planarity': 'first',
                    'rmsd_deviation': ['max', 'mean', 'count']
                })
                mol_summary.columns = ['pose_planarity', 'max_rmsd', 'mean_rmsd', 'num_rings']
                
                # Sort by pose planarity (worst first) and then by max_rmsd
                planarity_order = {'violation': 0, 'borderline': 1, 'acceptable': 2, 'excellent': 3}
                mol_summary['planarity_rank'] = mol_summary['pose_planarity'].map(planarity_order)
                mol_summary = mol_summary.sort_values(['planarity_rank', 'max_rmsd'], ascending=[True, False])
                mol_summary = mol_summary.drop('planarity_rank', axis=1)
                
                print(mol_summary.head(10))
    
    # Save CSV if requested
    if output_csv and not df.empty:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nResults saved to: {output_csv}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Analyze aromatic ring planarity in 3D molecular poses using tiered thresholds'
    )
    parser.add_argument('sdf_file', help='Input SDF file with 3D coordinates')
    parser.add_argument('-o', '--output', help='Output CSV file for results')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress detailed output')
    parser.add_argument('--write-violations', help='Write molecules with violations (>0.08Å) to SDF')
    parser.add_argument('--write-borderline', help='Write molecules with borderline cases (0.05-0.08Å) to SDF')
    parser.add_argument('--write-all-categories', action='store_true',
                        help='Write separate SDF files for each planarity category')
    parser.add_argument('--output-prefix', default='planarity_',
                        help='Prefix for SDF output files when using --write-all-categories')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.sdf_file).exists():
        print(f"Error: File '{args.sdf_file}' not found!")
        sys.exit(1)
    
    # Process SDF file
    df = process_sdf_file(
        args.sdf_file,
        output_csv=args.output,
        verbose=not args.quiet
    )
    
    # Skip SDF writing if dataframe is empty
    if df.empty:
        if not args.quiet:
            print("No aromatic rings found to analyze. No SDF files will be written.")
        return
    
    # Optionally write molecules with violations
    if args.write_violations:
        violating_mols = df[df['planarity_class'] == 'violation']['molecule'].unique()
        supplier = Chem.SDMolSupplier(args.sdf_file, removeHs=False)
        writer = Chem.SDWriter(args.write_violations)
        
        for mol in supplier:
            if mol and mol.HasProp('_Name') and mol.GetProp('_Name') in violating_mols:
                writer.write(mol)
        writer.close()
        
        if not args.quiet:
            print(f"\nMolecules with violations written to: {args.write_violations}")
    
    # Optionally write molecules with borderline planarity
    if args.write_borderline:
        borderline_mols = df[df['planarity_class'] == 'borderline']['molecule'].unique()
        supplier = Chem.SDMolSupplier(args.sdf_file, removeHs=False)
        writer = Chem.SDWriter(args.write_borderline)
        
        for mol in supplier:
            if mol and mol.HasProp('_Name') and mol.GetProp('_Name') in borderline_mols:
                writer.write(mol)
        writer.close()
        
        if not args.quiet:
            print(f"\nMolecules with borderline planarity written to: {args.write_borderline}")
    
    # Optionally write all categories to separate files
    if args.write_all_categories:
        # Get molecules for each planarity category based on pose_planarity
        categories = ['excellent', 'acceptable', 'borderline', 'violation']
        category_mols = {cat: df[df['pose_planarity'] == cat]['molecule'].unique() for cat in categories}
        
        supplier = Chem.SDMolSupplier(args.sdf_file, removeHs=False)
        
        for category, mol_list in category_mols.items():
            if len(mol_list) == 0:
                if not args.quiet:
                    print(f"\nNo molecules in category: {category}")
                continue
                
            output_file = f"{args.output_prefix}{category}.sdf"
            writer = Chem.SDWriter(output_file)
            
            mol_count = 0
            for mol in supplier:
                if mol and mol.HasProp('_Name') and mol.GetProp('_Name') in mol_list:
                    writer.write(mol)
                    mol_count += 1
            
            writer.close()
            
            if not args.quiet:
                print(f"\nWrote {mol_count} molecules with {category} planarity to: {output_file}")
            
            # Reset the supplier for the next category
            supplier = Chem.SDMolSupplier(args.sdf_file, removeHs=False)


if __name__ == "__main__":
    main()