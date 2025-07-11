import os
import sys
import glob
from rdkit import Chem

def check_and_convert_sdf_format(folder_path):
    # Find all SDF files in the folder
    sdf_files = glob.glob(os.path.join(folder_path, "*.sdf"))
    
    if not sdf_files:
        print(f"No SDF files found in {folder_path}")
        return
    
    print(f"Found {len(sdf_files)} SDF files in {folder_path}")
    
    for sdf_file in sdf_files:
        file_format = check_sdf_format(sdf_file)
        
        if file_format == "V2000":
            print(f"{os.path.basename(sdf_file)} is already in V2000 format")
        elif file_format == "V3000":
            print(f"{os.path.basename(sdf_file)} is in V3000 format - converting to V2000...")
            convert_to_v2000(sdf_file)
        else:
            print(f"{os.path.basename(sdf_file)} has unknown or mixed format - attempting to convert to V2000...")
            convert_to_v2000(sdf_file)

def check_sdf_format(sdf_file):
    try:
        with open(sdf_file, 'r') as f:
            content = f.read()
            
        # Split by molecule delimiter
        molecules = content.split("$$$$")
        
        # Check each molecule's format
        formats = []
        for mol in molecules:
            if mol.strip():  # Skip empty sections
                lines = mol.strip().split('\n')
                if len(lines) >= 4:  # Ensure we have enough lines
                    counts_line = lines[3]  # The 4th line is the counts line
                    if "V3000" in counts_line:
                        formats.append("V3000")
                    elif "V2000" in counts_line:
                        formats.append("V2000")
                    else:
                        formats.append("UNKNOWN")
        
        # If all molecules are the same format, return that format
        if formats and all(f == formats[0] for f in formats):
            return formats[0]
        else:
            return "MIXED"
    except Exception as e:
        print(f"Error checking format of {sdf_file}: {e}")
        return "ERROR"

def convert_to_v2000(sdf_file):
    try:
        # Create a backup of the original file
        backup_file = sdf_file + ".backup"
        os.rename(sdf_file, backup_file)
        
        # Read the molecules
        suppl = Chem.SDMolSupplier(backup_file, removeHs=False)
        
        # Write in V2000 format
        with Chem.SDWriter(sdf_file) as writer:
            writer.SetForceV3000(False)  # Ensure V2000 format
            for mol in suppl:
                if mol is not None:
                    writer.write(mol)
        
        print(f"Successfully converted {os.path.basename(sdf_file)} to V2000 format")
    except Exception as e:
        print(f"Error converting {sdf_file}: {e}")
        # Try to restore from backup if there was an error
        if os.path.exists(backup_file):
            os.rename(backup_file, sdf_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_sdf_to_v2000.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        sys.exit(1)
    
    check_and_convert_sdf_format(folder_path)
