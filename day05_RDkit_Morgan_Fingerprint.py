from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
smiles_list=[
    "CCO",
    "CCC",
    "CCN",
    "CCC1",
    "CC(=O)O",
    "HELLO"
]
fpgen=rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024
)
fingerprints=[]
valid_smiles=[]
for smiles in smiles_list:
    mol=Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp=fpgen.GetFingerprint(mol)
        fp_list=list(fp)
        fingerprints.append(fp_list)
        valid_smiles.append(smiles)
    else:
        print(f"Invalid SMILES skipped: {smiles}")
df_fp=pd.DataFrame(fingerprints)
df_fp.insert(0,"SMILES",valid_smiles)
activity=[1,0,1,1]
df_fp["Active"]=activity
print("\n Fingerprint DataFrame:\n")
print(df_fp.head())
print("\n Shape of DataFrame:")
print(df_fp.shape)
X=df_fp.drop(["SMILES", "Active"],axis=1)
y=df_fp["Active"]
print("\n Feature Matrix Shape:")
print(X.shape)
print("\n Target Shape:")
print(y.shape)

print("\n First 20 bits of first molecule:")
print(X.iloc[0,:20].tolist())
print("\n Number of ON bits:")
print(sum(X.iloc[0]))
print("\n Indices ofON bits:")
on_bits=[i for i,bit in enumerate(X.iloc[0]) if bit==1]
print(on_bits)
