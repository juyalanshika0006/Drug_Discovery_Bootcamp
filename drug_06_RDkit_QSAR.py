from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Creating dataset
smiles_list=[
    "CCO",
    "CCC",
    "CCN",
    "CCC1",
    "CC(=O)O",
    "HELLO"
]
#Activity labels
Activity=[1,0,1,1]

#Morgan fingerprint generator 
fpgen=rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=1024
)

# Create Empty lists
valid_smiles=[]
valid_activity=[]
fingerprints=[]

#Generate fingerprints
activity_index=0
for smiles in smiles_list:
    mol=Chem.MolFromSmiles(smiles)
    if mol is not None:
     fp=fpgen.GetFingerprint(mol)
     fp_list=list(fp)
     fingerprints.append(fp_list)
     valid_smiles.append(smiles)
     valid_activity.append(Activity[activity_index])
     activity_index+=1
    else:
       print(f"Invalid SMILES Ignored:\n")
#creating the dataframe 
df_fp=pd.DataFrame(fingerprints)
#add smiles in the first column 
df_fp.insert(0,"SMILES",valid_smiles)
df_fp["Active"]=valid_activity
print("\n====================")
print("Fingerprint DataFrame")
print("================\n")
print(df_fp.head())

#prepare for X and Y
X = df_fp.drop(["SMILES", "Active"], axis=1)

y = df_fp["Active"]

print("\nFeature Matrix Shape :", X.shape)

print("Target Shape :", y.shape)
 
#Split the Data
X_train,X_test,y_train,y_test=train_test_split(
  X,
  y,
  test_size=0.25,
  random_state=42
)
print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)
#create Model
model=RandomForestClassifier()
#train model
model.fit(X_train,y_train)
print("\n Model has learned from training model")
#Create predictions
predictions=model.predict(X_test)
print("\n predictions:")
print(predictions)

#Create Accuracy
accuracy=accuracy_score(y_test,predictions)
print("\n Accuracy:")
print(accuracy)

#Inspect  the molecule 
print("\n First 20 bits of first molecule:")
print(X.iloc[0,:20].tolist())
print("\n Number of ON bits:")
print(sum(X.iloc[0]))
print("\n Indices ofON bits:")
on_bits=[i for i,bit in enumerate(X.iloc[0]) if bit==1]
print(on_bits)
