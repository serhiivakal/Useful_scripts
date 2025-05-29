#!/usr/bin/env python
"""
Parallel SDF Merger for Target-Based Folder Structure
===================================================

This script processes a specific folder structure containing ligand SDF files
and merges them into a single file with appropriate source information,
utilizing multiprocessing for significantly improved performance.

Expected folder structure:
Target_folder/target_CHEMBLXXX/CHEMBLYYY/ligand.sdf

Purpose:
--------
- Process specific hierarchical folder structure of chemical data
- Virtually rename molecules based on their parent folder names
- Merge all found SDF files into a single consolidated file
- Utilize parallel processing for high performance with large datasets
- Provide visual progress indication during processing

Technical details:
-----------------
- Uses glob for specific pattern matching 
- Uses multiprocessing for parallel file processing
- Uses RDKit for SDF file handling
- Uses tqdm for progress visualization
- Adds properties to each molecule indicating source information
- Output file is named "merged_compounds.sdf" in the current directory

Usage:
------
python merge_target_sdf_parallel.py <target_directory> [--processes N]

Output:
-------
- merged_compounds.sdf: Combined file containing all molecules with source information

Requirements:
------------
- RDKit
- tqdm
- multiprocessing

Author: Serhii Vakal, Orion Pharma, Turku, Finland
Date: May 2025
"""

import os
import sys
import glob
import time
import argparse
from tqdm import tqdm
from rdkit import Chem
import multiprocessing as mp
from functools import partial
import tempfile


def find_target_dirs(target_directory):
    """Find all target directories."""
    target_pattern = os.path.join(target_directory, "target_*")
    return glob.glob(target_pattern)


def scan_target_dir(target_dir):
    """Scan a single target directory for SDF files."""
    target_id = os.path.basename(target_dir)
    result = []
    
    # Check each potential ligand directory
    ligand_dirs = glob.glob(os.path.join(target_dir, "*"))
    
    for ligand_dir in ligand_dirs:
        ligand_id = os.path.basename(ligand_dir)
        sdf_path = os.path.join(ligand_dir, "ligand.sdf")
        
        # Check if the file exists
        if os.path.isfile(sdf_path):
            result.append((sdf_path, target_id, ligand_id))
    
    return result


def find_sdf_files_parallel(target_directory, num_processes):
    """Find all SDF files using parallel processing."""
    # Find all target directories
    target_dirs = find_target_dirs(target_directory)
    print(f"Found {len(target_dirs)} target directories to scan...")
    
    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)
    
    # Process target directories in parallel with progress bar
    print(f"Scanning for SDF files using {num_processes} processes...")
    results = []
    
    with tqdm(total=len(target_dirs), desc="Scanning target directories") as pbar:
        for result in pool.imap_unordered(scan_target_dir, target_dirs):
            results.extend(result)
            pbar.update(1)
            pbar.set_postfix({"files_found": len(results)})
    
    pool.close()
    pool.join()
    
    return results


def process_sdf_batch(batch, output_file):
    """Process a batch of SDF files and write to a temporary output file."""
    writer = Chem.SDWriter(output_file)
    molecule_count = 0
    
    for file_path, target_id, ligand_id in batch:
        try:
            supplier = Chem.SDMolSupplier(file_path)
            
            for mol in supplier:
                if mol is not None:
                    # Add source information as properties
                    mol.SetProp("TargetID", target_id)
                    mol.SetProp("LigandID", ligand_id)
                    mol.SetProp("SourcePath", file_path)
                    
                    writer.write(mol)
                    molecule_count += 1
        except Exception as e:
            pass  # Silently handle errors in parallel processing
    
    writer.close()
    return molecule_count, output_file


def merge_sdf_files_parallel(sdf_file_info, output_file="merged_compounds.sdf", num_processes=None):
    """Merge SDF files using parallel processing."""
    if not sdf_file_info:
        print("No SDF files found to merge.")
        return 0
    
    # Use all available cores if not specified
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Divide files into batches for parallel processing
    batch_size = max(1, len(sdf_file_info) // (num_processes * 2))  # Ensure at least 2 batches per process
    batches = [sdf_file_info[i:i + batch_size] for i in range(0, len(sdf_file_info), batch_size)]
    
    print(f"Processing {len(sdf_file_info)} SDF files in {len(batches)} batches using {num_processes} processes...")
    
    # Create temporary output files for each batch
    temp_files = [tempfile.NamedTemporaryFile(suffix='.sdf', delete=False).name for _ in range(len(batches))]
    
    # Create a pool of workers
    pool = mp.Pool(processes=num_processes)
    
    # Process batches in parallel with progress bar
    total_molecules = 0
    results = []
    
    with tqdm(total=len(batches), desc="Processing batches") as pbar:
        for batch_idx, batch in enumerate(batches):
            results.append(pool.apply_async(
                process_sdf_batch, 
                args=(batch, temp_files[batch_idx]),
                callback=lambda x: pbar.update(1)
            ))
        
        # Wait for all processes to complete
        pool.close()
        pool.join()
    
    # Combine results
    batch_results = [r.get() for r in results]
    total_molecules = sum(count for count, _ in batch_results)
    
    # Merge all temporary files into the final output file
    print(f"Merging {len(temp_files)} temporary files into final output...")
    with open(output_file, 'wb') as outfile:
        for _, temp_file in batch_results:
            with open(temp_file, 'rb') as infile:
                outfile.write(infile.read())
            # Clean up temporary file
            os.unlink(temp_file)
    
    print(f"Merge complete! Successfully merged {total_molecules} molecules into {output_file}")
    print(f"Molecules were sourced from {len(sdf_file_info)} SDF files")
    
    return total_molecules


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Merge SDF files from a target-based folder structure in parallel.")
    parser.add_argument("target_directory", help="Directory containing the target folders")
    parser.add_argument("--processes", type=int, default=None, 
                        help="Number of parallel processes to use (default: all available cores)")
    parser.add_argument("--output", default="merged_compounds.sdf",
                        help="Output SDF file name (default: merged_compounds.sdf)")
    
    args = parser.parse_args()
    
    # Set number of processes
    num_processes = args.processes
    if num_processes is None:
        num_processes = mp.cpu_count()
        print(f"Using all available CPU cores: {num_processes}")
    
    # Check if directory exists
    if not os.path.isdir(args.target_directory):
        print(f"Error: Directory '{args.target_directory}' does not exist.")
        sys.exit(1)
    
    # Start time
    start_time = time.time()
    
    # Find all SDF files with parallel processing
    sdf_file_info = find_sdf_files_parallel(args.target_directory, num_processes)
    
    scan_time = time.time() - start_time
    print(f"Scanning completed in {scan_time:.1f} seconds.")
    
    if not sdf_file_info:
        print(f"No SDF files found in '{args.target_directory}' matching the expected pattern.")
        sys.exit(1)
    
    # Group files by target
    target_count = len(set([info[1] for info in sdf_file_info]))
    print(f"Found {len(sdf_file_info)} SDF files across {target_count} targets.")
    
    # Merge the SDF files using parallel processing
    merge_sdf_files_parallel(sdf_file_info, args.output, num_processes)
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds.")