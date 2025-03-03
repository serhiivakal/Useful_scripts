import os
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import MolWt
from tqdm import tqdm  # For progress bar

# List of common metals, ions, solvents, cofactors, and monosaccharides to exclude
EXCLUDE_RESIDUES = {
    # Metals and ions
    "NA", "K", "CA", "MG", "ZN", "FE", "CU", "MN", "CO", 
    "NI", "HG", "CD", "CL", "BR", "F", "I", "LI", "AL",
    "SR", "BA", "CR", "PB", "AG", "AU", "PT", "RB", "CS",
    "TI", "V", "MO", "RU", "RH", "PD", "OS", "IR", "W", "SE",
    # Solvents
    "HOH", "H2O", "DOD", "D2O", "SO4", "PO4", "NH4", "NO3",
    "CO3", "OXY", "OX", "ACT", "ACE", "ACN", "DMS", "DMSO",
    "IPA", "EOH", "EDO", "THF", "FMT", "GOL", "PEG", "PG4",
    # Cofactors and common small molecules
    "ATP", "ADP", "AMP", "FAD", "FADH", "FADH2", "NAD", "NADH",
    "NADP", "NADPH", "FMN", "HEM", "HEME", "PLP", "SAM", "COA",
    "TPP", "NAP", "NAPD", "NAPDH", "PMP", "GDP", "GTP", "CDP",
    "CTP", "UDP", "UMP", "TTP", "TMP", "UTP", "ITP", "DTT", "BME",
    # Common monosaccharides (adding here)
    "GLC", "GAL", "MAN", "FRU", "XYL", "RIB", "ARA", "FUC", "GLA", "IDR",
    "ALT", "ALL", "TAL", "SED", "KDO", "G6P", "G1P", "BGC", "BMA", "NAG"
    # Miscellaneous
    "SO4", "PO4", "CA2", "MG2", "MN3", "ZN2"
}

def is_unwanted_molecule(residue):
    """
    Check if a residue is an unwanted molecule (metal, ion, solvent, cofactor, or monosaccharide)
    based on its residue name or atom count.

    Args:
        residue (Bio.PDB.Residue): A residue object from BioPython.

    Returns:
        bool: True if the residue is unwanted, False otherwise.
    """
    # Check if residue name is in the exclusion list
    if residue.resname.strip().upper() in EXCLUDE_RESIDUES:
        return True

    # Exclude small entities with very few atoms (likely ions or small molecules)
    if len(list(residue.get_atoms())) <= 3:  # Usually ions or small junk molecules have 1-3 atoms
        return True

    return False

def is_valid_molecular_weight(mol):
    """
    Check if a molecule has a molecular weight within the desired range (100–900 Da).

    Args:
        mol (rdkit.Chem.Mol): An RDKit molecule object.

    Returns:
        bool: True if the molecular weight is within range, False otherwise.
    """
    if mol is None:
        return False

    # Calculate molecular weight
    mol_weight = MolWt(mol)

    # Check if within range
    return 100.0 <= mol_weight <= 900.0

def extract_ligands_to_sdf(pdb_folder):
    """
    Extract heteromolecules (ligands) from PDB files in the specified folder,
    filter out unwanted molecules, filter by molecular weight, and save each ligand
    as a separate SDF file with the original PDB filename stored as a property.

    Args:
        pdb_folder (str): Path to the folder containing PDB files.
    """
    # Create an output folder for SDF files
    sdf_output_folder = "sdf_output"
    os.makedirs(sdf_output_folder, exist_ok=True)

    # Initialize BioPython's PDB parser
    parser = PDBParser(QUIET=True)

    # Count the number of PDB files in the folder
    pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith(".pdb")]
    total_files = len(pdb_files)
    print(f"Found {total_files} PDB files to process.")

    # Progress bar using tqdm
    with tqdm(total=total_files, desc="Processing PDB files") as pbar:
        # Process each PDB file in the folder
        for pdb_file in pdb_files:
            pdb_path = os.path.join(pdb_folder, pdb_file)

            # Parse the PDB structure
            try:
                structure = parser.get_structure(pdb_file, pdb_path)
            except Exception as e:
                print(f"\nError parsing {pdb_file}: {e}")
                pbar.update(1)
                continue

            # Extract heteromolecules (ligands)
            for model in structure:
                for chain in model:
                    for residue in chain:
                        # Skip standard residues (proteins, DNA, water, etc.)
                        if residue.id[0] == " ":
                            continue

                        # Skip unwanted molecules based on exclusion criteria
                        if is_unwanted_molecule(residue):
                            continue

                        # Extract the ligand
                        ligand_id = residue.resname
                        ligand_atoms = list(residue.get_atoms())

                        # Save the ligand to a temporary PDB file
                        ligand_pdb_path = os.path.join(sdf_output_folder, f"{ligand_id}_{pdb_file}.pdb")
                        io = PDBIO()
                        io.set_structure(residue)
                        io.save(ligand_pdb_path)

                        # Convert the ligand to SDF using RDKit
                        try:
                            # Read the PDB file with RDKit
                            mol = Chem.MolFromPDBFile(ligand_pdb_path, removeHs=False)
                            if mol is None:
                                print(f"Failed to parse ligand {ligand_id} from {pdb_file}")
                                continue

                            # Filter by molecular weight
                            if not is_valid_molecular_weight(mol):
                                print(f"Filtered out {ligand_id} from {pdb_file} due to molecular weight")
                                continue

                            # Add the original PDB filename as a property
                            mol.SetProp("OriginalPDBFile", pdb_file)

                            # Generate 3D coordinates if missing
                            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

                            # Write to SDF file
                            sdf_file = os.path.join(sdf_output_folder, f"{ligand_id}_{pdb_file[:-4]}.sdf")
                            writer = Chem.SDWriter(sdf_file)
                            writer.write(mol)
                            writer.close()

                        except Exception as e:
                            print(f"Error processing ligand {ligand_id} from {pdb_file}: {e}")
                        finally:
                            # Clean up temporary files
                            if os.path.exists(ligand_pdb_path):
                                os.remove(ligand_pdb_path)

            # Update the progress bar
            pbar.update(1)

if __name__ == "__main__":
    # Specify the folder containing PDB files (current directory in this case)
    pdb_folder = "."
    extract_ligands_to_sdf(pdb_folder)