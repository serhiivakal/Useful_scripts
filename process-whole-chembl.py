from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import pandas as pd
from tqdm import tqdm
import numpy as np

def convert_sdf_to_csv(sdf_file, output_csv):
    """Convert SDF to CSV with essential information"""
    
    # Initialize empty lists for data
    data = {
        'SMILES': [],
        'ChEMBL_ID': [],
        'Bioactivity_Value': [],
        'Activity_Type': []
    }
    
    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_file)
    
    # Process molecules with progress bar
    for mol in tqdm(supplier, desc="Processing molecules"):
        if mol is not None:
            try:
                # Generate canonical SMILES
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                
                # Get properties
                chembl_id = mol.GetProp('chembl_id') if mol.HasProp('chembl_id') else ''
                bioactivity = mol.GetProp('bioactivity_value') if mol.HasProp('bioactivity_value') else ''
                activity_type = mol.GetProp('activity_type') if mol.HasProp('activity_type') else ''
                
                # Append to data dictionary
                data['SMILES'].append(smiles)
                data['ChEMBL_ID'].append(chembl_id)
                data['Bioactivity_Value'].append(bioactivity)
                data['Activity_Type'].append(activity_type)
                
            except Exception as e:
                print(f"Error processing molecule: {e}")
                continue
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Successfully converted {len(df)} compounds to CSV")
    return df

def search_similar_compounds(query_smiles, csv_file, similarity_threshold=0.8):
    """Search for similar compounds in the CSV database"""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create query molecule fingerprint
    query_mol = Chem.MolFromSmiles(query_smiles)
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, 1024)
    
    # Function to calculate similarity
    def calculate_similarity(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            return DataStructs.TanimotoSimilarity(query_fp, fp)
        except:
            return 0
    
    # Calculate similarities
    df['Similarity'] = df['SMILES'].apply(calculate_similarity)
    
    # Filter by similarity threshold
    similar_compounds = df[df['Similarity'] >= similarity_threshold].sort_values(
        by='Similarity', ascending=False
    )
    
    return similar_compounds

# Example usage
if __name__ == "__main__":
    # First convert SDF to CSV (run once)
    sdf_file = "chembl_database.sdf"
    csv_file = "chembl_database.csv"
    df = convert_sdf_to_csv(sdf_file, csv_file)
    
    # Search for similar compounds (can be run multiple times)
    query_smiles = "CC1=CC=C(C=C1)NC(=O)C2=CC=CS2"  # Example SMILES
    results = search_similar_compounds(query_smiles, csv_file, similarity_threshold=0.7)
    print(f"Found {len(results)} similar compounds")