from Bio import PDB
from pathlib import Path
import pandas as pd
import os

def parse_uniprot_chembl_mapping(mapping_file):
    """Parse the UniProt-ChEMBL mapping file"""
    data = {
        'UniProt_ID': [],
        'ChEMBL_ID': [],
        'Target_Name': [],
        'Target_Type': []
    }
    
    with open(mapping_file, 'r') as f:
        # Skip header line
        next(line for line in f if line.startswith('#'))
        
        for line in f:
            if line.strip():
                fields = line.strip().split('\t')
                data['UniProt_ID'].append(fields[0])
                data['ChEMBL_ID'].append(fields[1])
                data['Target_Name'].append(fields[2])
                data['Target_Type'].append(fields[3])
    
    return pd.DataFrame(data)

def extract_uniprot_from_pdb(pdb_file):
    """Extract UniProt ID from PDB file"""
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('DBREF'):
                    fields = line.split()
                    if 'UNP' in fields or 'UNIPROT' in fields:
                        # UniProt ID typically follows the database identifier
                        idx = fields.index('UNP' if 'UNP' in fields else 'UNIPROT')
                        return fields[idx + 1]
    except Exception as e:
        print(f"Error processing {pdb_file}: {str(e)}")
    return None

def process_pdb_files(pdb_folder, mapping_file, output_csv):
    """Process PDB files and create mapping"""
    # Read UniProt-ChEMBL mapping
    chembl_mapping = parse_uniprot_chembl_mapping(mapping_file)
    # Drop duplicates keeping the first occurrence for each UniProt_ID
    chembl_mapping = chembl_mapping.drop_duplicates(subset=['UniProt_ID'], keep='first')
    
    results = []
    pdb_files = []
    for ext in ['.pdb', '.ent']:
        pdb_files.extend(Path(pdb_folder).glob(f'*{ext}'))
    
    print(f"Processing {len(pdb_files)} PDB files...")
    
    for pdb_file in pdb_files:
        try:
            uniprot_id = extract_uniprot_from_pdb(str(pdb_file))
            if uniprot_id:
                # Match with ChEMBL mapping - now will only get one match
                chembl_match = chembl_mapping[chembl_mapping['UniProt_ID'] == uniprot_id]
                
                if not chembl_match.empty:
                    # Take only the first match
                    row = chembl_match.iloc[0]
                    results.append({
                        'PDB_File': pdb_file.name,
                        'UniProt_ID': uniprot_id,
                        'ChEMBL_ID': row['ChEMBL_ID'],
                        'Target_Name': row['Target_Name'],
                        'Target_Type': row['Target_Type']
                    })
                else:
                    results.append({
                        'PDB_File': pdb_file.name,
                        'UniProt_ID': uniprot_id,
                        'ChEMBL_ID': 'Not Found',
                        'Target_Name': 'Not Found',
                        'Target_Type': 'Not Found'
                    })
            else:
                print(f"No UniProt ID found for {pdb_file.name}")
                
        except Exception as e:
            print(f"Failed to process {pdb_file.name}: {str(e)}")
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    print(f"Successfully processed {len(pdb_files)} PDB files")
    print(f"Found {len(df)} unique matches")
    return df

if __name__ == "__main__":
    # Configure paths
    current_dir = os.getcwd()
    pdb_folder = current_dir  # Current directory, modify as needed
    mapping_file = os.path.join(current_dir, "chembl_uniprot_mapping.txt")
    output_csv = os.path.join(current_dir, "pdb_uniprot_chembl_matches.csv")
    
    # Process files and create mapping
    results_df = process_pdb_files(pdb_folder, mapping_file, output_csv)