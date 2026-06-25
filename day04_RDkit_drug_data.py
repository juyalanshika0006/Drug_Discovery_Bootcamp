#import libraries 
from rdkit import Chem 
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen

import pandas as pd
#list of moleecules
smiles_list = [
    "CCO",      # Ethanol
    "CCC",      # Propane
    "CCN",      # Ethylamine
    "HELLO"     # Invalid SMILES
]
#empty list for results
data=[]
#Process each SMILES
for smiles in smiles_list:
    #convert SMILES to Molecule
    mol=Chem.MolFromSmiles(smiles)
    #safe check
    if mol is not None:
         # Calculate Descriptors
        mw = Descriptors.MolWt(mol)

        logp = Crippen.MolLogP(mol)

        hba = Descriptors.NumHAcceptors(mol)

        hbd = Descriptors.NumHDonors(mol)

        tpsa = Descriptors.TPSA(mol)
        #store results
        data.append([
            smiles,
            mw,
            logp,
            hba,
            hbd,
            tpsa
        ])
    else:
        print(f"Invalid SMILES skipped:{smiles}")
#Create DataFrame        
df=pd.DataFrame(
    data,
    columns=[
        "SMILES",
        "MW",
        "LogP",
        "HBA",
        "HBD",
        "TPSA"
    ]
)        
#show Results
print("\n Descriptor Table:\n")
print(df)


# using the feature of Morganfingerprinting
from rdkit.Chem import rdFingerprintGenerator
mol=Chem.MolFromSmiles("CCO")
fpgen=rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024
)
fp=fpgen.GetFingerprint(mol)
fp_list=list(fp)
print(fp_list[:100])
