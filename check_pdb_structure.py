#!/usr/bin/env python3
"""
Comprehensive PDB structure checker for common issues and inconsistencies.
Usage: python check_pdb_structure.py input.pdb
"""

import sys
import os
import re
from collections import defaultdict, Counter
from math import sqrt

# Standard amino acid residues and their expected backbone atoms
STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
}

# Expected backbone atoms for all amino acids
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O'}

# Expected atoms for each amino acid (including backbone)
EXPECTED_ATOMS = {
    'ALA': {'N', 'CA', 'C', 'O', 'CB'},
    'ARG': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'},
    'ASN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2'},
    'ASP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2'},
    'CYS': {'N', 'CA', 'C', 'O', 'CB', 'SG'},
    'GLN': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2'},
    'GLU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2'},
    'GLY': {'N', 'CA', 'C', 'O'},
    'HIS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'},
    'ILE': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1'},
    'LEU': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2'},
    'LYS': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ'},
    'MET': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE'},
    'PHE': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'},
    'PRO': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD'},
    'SER': {'N', 'CA', 'C', 'O', 'CB', 'OG'},
    'THR': {'N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2'},
    'TRP': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'},
    'TYR': {'N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'},
    'VAL': {'N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2'}
}

class PDBChecker:
    def __init__(self, filename):
        self.filename = filename
        self.atoms = []
        self.residues = defaultdict(list)
        self.chains = defaultdict(list)
        self.issues = []
        self.warnings = []
        self.info = []
        self.zero_occupancy_atoms = []
        self.atom_serial_gaps = []
        self.residue_gaps = defaultdict(list)
        
    def parse_pdb(self):
        """Parse PDB file and extract atom information."""
        try:
            with open(self.filename, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            self.issues.append(f"Cannot read file: {e}")
            return False
            
        for line_num, line in enumerate(lines, 1):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_data = {
                        'line_num': line_num,
                        'record_type': line[0:6].strip(),
                        'atom_serial': int(line[6:11].strip()) if line[6:11].strip().isdigit() else None,
                        'atom_name': line[12:16].strip(),
                        'residue_name': line[17:20].strip(),
                        'chain_id': line[21:22].strip(),
                        'residue_number': line[22:26].strip(),
                        'insertion_code': line[26:27].strip(),
                        'x': float(line[30:38].strip()) if line[30:38].strip() else None,
                        'y': float(line[38:46].strip()) if line[38:46].strip() else None,
                        'z': float(line[46:54].strip()) if line[46:54].strip() else None,
                        'occupancy': float(line[54:60].strip()) if line[54:60].strip() else 1.0,
                        'b_factor': float(line[60:66].strip()) if line[60:66].strip() else 0.0,
                        'element': line[76:78].strip() if len(line) > 77 else '',
                        'full_line': line.rstrip()
                    }
                    
                    # Convert residue number to integer if possible
                    try:
                        atom_data['residue_number_int'] = int(atom_data['residue_number'])
                    except ValueError:
                        atom_data['residue_number_int'] = None
                        
                    self.atoms.append(atom_data)
                    
                    # Group by residue and chain
                    residue_key = f"{atom_data['chain_id']}_{atom_data['residue_number']}_{atom_data['insertion_code']}"
                    self.residues[residue_key].append(atom_data)
                    self.chains[atom_data['chain_id']].append(atom_data)
                    
                except (ValueError, IndexError) as e:
                    self.issues.append(f"Line {line_num}: Malformed PDB line - {e}")
                    
        return True
        
    def check_atom_numbering(self):
        """Check for issues with atom serial numbering."""
        if not self.atoms:
            return
            
        serials = [atom['atom_serial'] for atom in self.atoms if atom['atom_serial'] is not None]
        
        if not serials:
            self.issues.append("No valid atom serial numbers found")
            return
            
        # Check for duplicates
        serial_counts = Counter(serials)
        duplicates = [serial for serial, count in serial_counts.items() if count > 1]
        if duplicates:
            self.issues.append(f"Duplicate atom serial numbers: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}")
            
        # Check for sequential numbering
        if serials != sorted(serials):
            self.warnings.append("Atom serial numbers are not in sequential order")
            
        # Check for gaps and store detailed information
        if serials:
            sorted_serials = sorted(serials)
            expected_range = list(range(min(sorted_serials), max(sorted_serials) + 1))
            missing = set(expected_range) - set(serials)
            
            if missing:
                self.warnings.append(f"Gaps in atom serial numbering: {len(missing)} missing numbers")
                
                # Group consecutive missing numbers into ranges
                missing_sorted = sorted(missing)
                gap_ranges = []
                start = missing_sorted[0]
                end = missing_sorted[0]
                
                for i in range(1, len(missing_sorted)):
                    if missing_sorted[i] == end + 1:
                        end = missing_sorted[i]
                    else:
                        if start == end:
                            gap_ranges.append(str(start))
                        else:
                            gap_ranges.append(f"{start}-{end}")
                        start = missing_sorted[i]
                        end = missing_sorted[i]
                
                # Add the last range
                if start == end:
                    gap_ranges.append(str(start))
                else:
                    gap_ranges.append(f"{start}-{end}")
                    
                self.atom_serial_gaps = gap_ranges
                
    def check_residue_numbering(self):
        """Check for issues with residue numbering."""
        for chain_id, chain_atoms in self.chains.items():
            residue_numbers = []
            residue_data = {}
            
            for atom in chain_atoms:
                res_key = (atom['residue_number_int'], atom['insertion_code'])
                if res_key not in residue_numbers:
                    residue_numbers.append(res_key)
                    residue_data[res_key] = {
                        'residue_name': atom['residue_name'],
                        'residue_number': atom['residue_number'],
                        'insertion_code': atom['insertion_code']
                    }
                    
            # Check for sequential numbering within chain
            int_residues = [r[0] for r in residue_numbers if r[0] is not None]
            if int_residues and int_residues != sorted(int_residues):
                self.warnings.append(f"Chain {chain_id}: Residue numbers not in sequential order")
                
            # Check for large gaps and store detailed information
            if len(int_residues) > 1:
                sorted_residues = sorted(int_residues)
                chain_gaps = []
                
                for i in range(1, len(sorted_residues)):
                    gap_size = sorted_residues[i] - sorted_residues[i-1] - 1
                    if gap_size > 0:
                        gap_info = {
                            'start_residue': sorted_residues[i-1],
                            'end_residue': sorted_residues[i],
                            'gap_size': gap_size,
                            'missing_residues': list(range(sorted_residues[i-1] + 1, sorted_residues[i]))
                        }
                        chain_gaps.append(gap_info)
                        
                        if gap_size > 10:  # Large gap threshold
                            self.warnings.append(f"Chain {chain_id}: Large gap in residue numbering between {sorted_residues[i-1]} and {sorted_residues[i]} (missing {gap_size} residues)")
                
                if chain_gaps:
                    self.residue_gaps[chain_id] = chain_gaps
                    
    def check_missing_atoms(self):
        """Check for missing atoms in standard amino acids."""
        for residue_key, atoms in self.residues.items():
            if not atoms:
                continue
                
            residue_name = atoms[0]['residue_name']
            chain_id = atoms[0]['chain_id']
            residue_number = atoms[0]['residue_number']
            
            if residue_name in STANDARD_AA:
                present_atoms = {atom['atom_name'] for atom in atoms}
                expected_atoms = EXPECTED_ATOMS.get(residue_name, set())
                
                # Check for missing backbone atoms
                missing_backbone = BACKBONE_ATOMS - present_atoms
                if missing_backbone:
                    self.issues.append(f"Chain {chain_id}, residue {residue_name} {residue_number}: Missing backbone atoms: {missing_backbone}")
                    
                # Check for missing side chain atoms
                missing_sidechain = expected_atoms - present_atoms - missing_backbone
                if missing_sidechain:
                    self.warnings.append(f"Chain {chain_id}, residue {residue_name} {residue_number}: Missing side chain atoms: {missing_sidechain}")
                    
                # Check for unexpected atoms
                unexpected_atoms = present_atoms - expected_atoms
                if unexpected_atoms:
                    # Filter out common modifications (hydrogens, alternate conformations)
                    unexpected_filtered = {atom for atom in unexpected_atoms 
                                         if not atom.startswith('H') and not atom.endswith(('A', 'B'))}
                    if unexpected_filtered:
                        self.warnings.append(f"Chain {chain_id}, residue {residue_name} {residue_number}: Unexpected atoms: {unexpected_filtered}")
                        
    def check_coordinates(self):
        """Check for coordinate-related issues."""
        coords_issues = []
        
        for atom in self.atoms:
            # Check for missing coordinates
            if atom['x'] is None or atom['y'] is None or atom['z'] is None:
                coords_issues.append(f"Line {atom['line_num']}: Missing coordinates")
                continue
                
            # Check for unreasonable coordinates (very large values)
            coords = [atom['x'], atom['y'], atom['z']]
            if any(abs(coord) > 9999 for coord in coords):
                coords_issues.append(f"Line {atom['line_num']}: Extremely large coordinates: ({atom['x']:.2f}, {atom['y']:.2f}, {atom['z']:.2f})")
                
        if coords_issues:
            if len(coords_issues) > 10:
                self.issues.extend(coords_issues[:10])
                self.issues.append(f"... and {len(coords_issues) - 10} more coordinate issues")
            else:
                self.issues.extend(coords_issues)
                
    def check_duplicate_atoms(self):
        """Check for duplicate atoms (same residue, same atom name)."""
        atom_positions = defaultdict(list)
        
        for atom in self.atoms:
            key = (atom['chain_id'], atom['residue_number'], atom['insertion_code'], atom['atom_name'])
            atom_positions[key].append(atom)
            
        duplicates = {key: atoms for key, atoms in atom_positions.items() if len(atoms) > 1}
        
        for key, atoms in list(duplicates.items())[:10]:  # Limit output
            chain, res_num, ins_code, atom_name = key
            self.warnings.append(f"Duplicate atom: Chain {chain}, residue {res_num}{ins_code}, atom {atom_name} (appears {len(atoms)} times)")
            
        if len(duplicates) > 10:
            self.warnings.append(f"... and {len(duplicates) - 10} more duplicate atom issues")
            
    def check_chain_breaks(self):
        """Check for potential chain breaks based on CA-CA distances."""
        for chain_id, chain_atoms in self.chains.items():
            # Get CA atoms sorted by residue number
            ca_atoms = [atom for atom in chain_atoms if atom['atom_name'] == 'CA']
            ca_atoms.sort(key=lambda x: (x['residue_number_int'] or float('inf'), x['insertion_code']))
            
            for i in range(len(ca_atoms) - 1):
                atom1, atom2 = ca_atoms[i], ca_atoms[i + 1]
                
                if atom1['x'] is None or atom2['x'] is None:
                    continue
                    
                # Calculate distance
                dx = atom2['x'] - atom1['x']
                dy = atom2['y'] - atom1['y']
                dz = atom2['z'] - atom1['z']
                distance = sqrt(dx*dx + dy*dy + dz*dz)
                
                # Typical CA-CA distance is ~3.8Å, chain break if > 5Å
                if distance > 5.0:
                    self.warnings.append(f"Chain {chain_id}: Potential chain break between residues {atom1['residue_number']} and {atom2['residue_number']} (CA-CA distance: {distance:.2f}Å)")
                elif distance > 4.5:
                    self.info.append(f"Chain {chain_id}: Large CA-CA distance between residues {atom1['residue_number']} and {atom2['residue_number']} ({distance:.2f}Å)")
                    
    def check_occupancy_and_bfactor(self):
        """Check for unusual occupancy and B-factor values."""
        occupancies = [atom['occupancy'] for atom in self.atoms if atom['occupancy'] is not None]
        b_factors = [atom['b_factor'] for atom in self.atoms if atom['b_factor'] is not None]
        
        # Find atoms with zero occupancy
        for atom in self.atoms:
            if atom['occupancy'] == 0.0:
                self.zero_occupancy_atoms.append({
                    'line_num': atom['line_num'],
                    'atom_serial': atom['atom_serial'],
                    'atom_name': atom['atom_name'],
                    'residue_name': atom['residue_name'],
                    'chain_id': atom['chain_id'],
                    'residue_number': atom['residue_number'],
                    'insertion_code': atom['insertion_code']
                })
        
        # Check occupancy
        zero_occupancy = len(self.zero_occupancy_atoms)
        if zero_occupancy > 0:
            self.warnings.append(f"{zero_occupancy} atoms have zero occupancy")
            
        high_occupancy = sum(1 for occ in occupancies if occ > 1.0)
        if high_occupancy > 0:
            self.warnings.append(f"{high_occupancy} atoms have occupancy > 1.0")
            
        # Check B-factors
        if b_factors:
            avg_b = sum(b_factors) / len(b_factors)
            very_high_b = sum(1 for b in b_factors if b > 100.0)
            if very_high_b > 0:
                self.warnings.append(f"{very_high_b} atoms have very high B-factors (>100)")
            if avg_b > 50.0:
                self.info.append(f"Average B-factor is high: {avg_b:.2f}")
                
    def check_water_and_hetero(self):
        """Check water and hetero atom issues."""
        water_atoms = [atom for atom in self.atoms if atom['residue_name'] == 'HOH']
        hetero_atoms = [atom for atom in self.atoms if atom['record_type'] == 'HETATM' and atom['residue_name'] != 'HOH']
        
        if water_atoms:
            self.info.append(f"Found {len(water_atoms)} water atoms")
            
        if hetero_atoms:
            hetero_residues = set((atom['residue_name'] for atom in hetero_atoms))
            self.info.append(f"Found {len(hetero_atoms)} hetero atoms in {len(hetero_residues)} different residue types: {sorted(hetero_residues)}")
            
    def generate_summary(self):
        """Generate summary statistics."""
        if not self.atoms:
            return
            
        total_atoms = len(self.atoms)
        protein_atoms = len([atom for atom in self.atoms if atom['record_type'] == 'ATOM'])
        hetero_atoms = len([atom for atom in self.atoms if atom['record_type'] == 'HETATM'])
        
        chains = set(atom['chain_id'] for atom in self.atoms)
        residues = set((atom['chain_id'], atom['residue_number'], atom['insertion_code']) for atom in self.atoms)
        
        self.info.append(f"Total atoms: {total_atoms} (Protein: {protein_atoms}, Hetero: {hetero_atoms})")
        self.info.append(f"Chains: {len(chains)} ({sorted(chains)})")
        self.info.append(f"Residues: {len(residues)}")
        
    def print_zero_occupancy_atoms(self):
        """Print detailed list of atoms with zero occupancy."""
        if not self.zero_occupancy_atoms:
            return
            
        print(f"\n🔍 ATOMS WITH ZERO OCCUPANCY ({len(self.zero_occupancy_atoms)}):")
        print(f"{'Line':<6} {'Serial':<6} {'Chain':<5} {'Residue':<8} {'Atom':<4} {'ResNum':<6}")
        print("-" * 40)
        
        for atom in self.zero_occupancy_atoms:
            ins_code = atom['insertion_code'] if atom['insertion_code'] else ''
            print(f"{atom['line_num']:<6} {atom['atom_serial']:<6} {atom['chain_id']:<5} "
                  f"{atom['residue_name']:<8} {atom['atom_name']:<4} {atom['residue_number']}{ins_code:<6}")
                  
    def print_gaps(self):
        """Print detailed information about all gaps."""
        # Print atom serial number gaps
        if self.atom_serial_gaps:
            print(f"\n🔍 ATOM SERIAL NUMBER GAPS:")
            print(f"Missing atom serial numbers: {', '.join(self.atom_serial_gaps)}")
            
        # Print residue number gaps
        if self.residue_gaps:
            print(f"\n🔍 RESIDUE NUMBER GAPS:")
            for chain_id, gaps in self.residue_gaps.items():
                print(f"\nChain {chain_id}:")
                for gap in gaps:
                    print(f"  Gap between residues {gap['start_residue']} and {gap['end_residue']}")
                    print(f"    Missing {gap['gap_size']} residues: {gap['missing_residues']}")
        
    def run_all_checks(self):
        """Run all checks and return results."""
        if not self.parse_pdb():
            return False
            
        self.generate_summary()
        self.check_atom_numbering()
        self.check_residue_numbering()
        self.check_missing_atoms()
        self.check_coordinates()
        self.check_duplicate_atoms()
        self.check_chain_breaks()
        self.check_occupancy_and_bfactor()
        self.check_water_and_hetero()
        
        return True
        
    def print_results(self):
        """Print all findings."""
        print(f"\n{'='*60}")
        print(f"PDB Structure Analysis: {os.path.basename(self.filename)}")
        print(f"{'='*60}")
        
        if self.info:
            print(f"\n📊 SUMMARY:")
            for item in self.info:
                print(f"   {item}")
                
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")
                
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
                
        # Print detailed information about zero occupancy atoms and gaps
        self.print_zero_occupancy_atoms()
        self.print_gaps()
                
        if not self.issues and not self.warnings:
            print(f"\n✅ No major issues found!")
            
        print(f"\n{'='*60}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_pdb_structure.py input.pdb")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
        
    checker = PDBChecker(input_file)
    
    if checker.run_all_checks():
        checker.print_results()
    else:
        print("Failed to analyze PDB file.")
        sys.exit(1)

if __name__ == "__main__":
    main()