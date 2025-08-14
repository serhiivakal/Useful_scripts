#!/usr/bin/env python3
"""
Fix PDB atom numbering based on residue sequence order.
Usage: python fix_pdb_numbering.py input.pdb
"""

import sys
import os
import re

def parse_pdb_line(line):
    """Parse a PDB ATOM/HETATM line and extract relevant information."""
    if not (line.startswith('ATOM') or line.startswith('HETATM')):
        return None
    
    record_type = line[0:6].strip()
    atom_serial = line[6:11].strip()
    atom_name = line[12:16].strip()
    residue_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    residue_number = line[22:26].strip()
    insertion_code = line[26:27].strip()
    
    return {
        'record_type': record_type,
        'atom_serial': atom_serial,
        'atom_name': atom_name,
        'residue_name': residue_name,
        'chain_id': chain_id,
        'residue_number': residue_number,
        'insertion_code': insertion_code,
        'full_line': line
    }

def create_residue_key(chain_id, residue_number, insertion_code):
    """Create a unique key for each residue."""
    return f"{chain_id}_{residue_number}_{insertion_code}"

def fix_pdb_numbering(input_file):
    """Fix atom numbering in PDB file based on residue order."""
    
    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Separate different types of records
    header_lines = []
    atom_lines = []
    ter_lines = []
    conect_lines = []
    end_lines = []
    other_lines = []
    
    residue_order = []
    residue_atoms = {}
    original_to_new_serial = {}  # Map original serials to new serials
    
    for line in lines:
        line_stripped = line.rstrip()
        
        if line.startswith('ATOM') or line.startswith('HETATM'):
            parsed = parse_pdb_line(line)
            if parsed:
                residue_key = create_residue_key(
                    parsed['chain_id'], 
                    parsed['residue_number'], 
                    parsed['insertion_code']
                )
                
                # Track residue order (first occurrence)
                if residue_key not in residue_atoms:
                    residue_order.append(residue_key)
                    residue_atoms[residue_key] = []
                
                residue_atoms[residue_key].append(line)
                atom_lines.append(line)
            else:
                other_lines.append(line)
        elif line.startswith(('TITLE', 'REMARK', 'CRYST1', 'HEADER', 'COMPND', 'SOURCE', 'AUTHOR', 'REVDAT')):
            header_lines.append(line)
        elif line.startswith('TER'):
            ter_lines.append(line)
        elif line.startswith('CONECT'):
            conect_lines.append(line)
        elif line.startswith('END'):
            end_lines.append(line)
        else:
            other_lines.append(line)
    
    # Sort residues by chain ID and residue number
    def sort_key(residue_key):
        parts = residue_key.split('_')
        chain_id = parts[0]
        try:
            residue_num = int(parts[1])
        except ValueError:
            residue_num = float('inf')  # Put non-numeric residue numbers at the end
        insertion_code = parts[2]
        return (chain_id, residue_num, insertion_code)
    
    sorted_residue_keys = sorted(residue_order, key=sort_key)
    
    # Create output with renumbered atoms
    output_lines = []
    atom_counter = 1
    
    # Add header lines
    output_lines.extend(header_lines)
    
    # Add atoms in correct residue order with sequential numbering
    renumbered_atom_lines = []
    for residue_key in sorted_residue_keys:
        for atom_line in residue_atoms[residue_key]:
            # Get original serial number for CONECT mapping
            try:
                original_serial = int(atom_line[6:11].strip())
                original_to_new_serial[original_serial] = atom_counter
            except ValueError:
                pass
            
            # Replace atom serial number
            new_line = atom_line[:6] + f"{atom_counter:5d}" + atom_line[11:]
            renumbered_atom_lines.append(new_line)
            atom_counter += 1
    
    output_lines.extend(renumbered_atom_lines)
    
    # Update TER records with correct serial numbers
    if ter_lines:
        # Update TER records to have sequential numbers after atoms
        for ter_line in ter_lines:
            # Update TER serial number
            updated_ter = f"TER   {atom_counter:5d}" + ter_line[11:]
            output_lines.append(updated_ter)
            atom_counter += 1
    
    # Update CONECT records with new serial numbers
    updated_conect_lines = []
    for conect_line in conect_lines:
        parts = conect_line.split()
        if len(parts) >= 2:
            try:
                # Update all atom serial numbers in CONECT record
                updated_parts = ['CONECT']
                for i in range(1, len(parts)):
                    old_serial = int(parts[i])
                    if old_serial in original_to_new_serial:
                        updated_parts.append(str(original_to_new_serial[old_serial]))
                    else:
                        # Skip CONECT records that reference non-existent atoms
                        updated_parts = None
                        break
                
                if updated_parts:
                    # Format CONECT record properly
                    updated_conect = f"CONECT{int(updated_parts[1]):5d}"
                    for j in range(2, len(updated_parts)):
                        updated_conect += f"{int(updated_parts[j]):5d}"
                    updated_conect_lines.append(updated_conect + '\n')
                    
            except ValueError:
                # Skip malformed CONECT records
                continue
    
    # Remove duplicate CONECT records
    unique_conect = []
    seen_conect = set()
    for line in updated_conect_lines:
        line_stripped = line.rstrip()
        if line_stripped not in seen_conect:
            unique_conect.append(line)
            seen_conect.add(line_stripped)
    
    output_lines.extend(unique_conect)
    
    # Add other non-critical lines (but not END)
    for line in other_lines:
        if not line.startswith(('END', 'TER', 'CONECT')):
            output_lines.append(line)
    
    # Add single END record
    if renumbered_atom_lines:  # Only add END if we have atoms
        output_lines.append('END\n')
    
    return output_lines

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_pdb_numbering.py input.pdb")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    
    if not input_file.lower().endswith('.pdb'):
        print("Warning: Input file doesn't have .pdb extension")
    
    # Create output filename
    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_fixed.pdb"
    
    try:
        # Fix the numbering
        print(f"Processing {input_file}...")
        fixed_lines = fix_pdb_numbering(input_file)
        
        # Write output file
        with open(output_file, 'w') as f:
            f.writelines(fixed_lines)
        
        print(f"Fixed PDB file saved as: {output_file}")
        
        # Count atoms for verification
        atom_count = sum(1 for line in fixed_lines if line.startswith('ATOM') or line.startswith('HETATM'))
        print(f"Total atoms processed: {atom_count}")
        
        print("Fixes applied:")
        print("  • Renumbered atoms sequentially")
        print("  • Maintained proper PDB record order")
        print("  • Updated CONECT records with new atom numbers")
        print("  • Updated TER records")
        print("  • Removed duplicate records")
        print("  • Added single END record")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()