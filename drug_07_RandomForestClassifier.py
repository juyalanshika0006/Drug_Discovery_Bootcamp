from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


smiles_list = [
    "CCO",
    "CCC",
    "CCN",
    "CCCl",
    "CCBr",
    "CCCO",
    "CCCN",
    "CC(=O)O",
    "c1ccccc1",
    "CCOC",
    "CCS",
    "CCF"
]

activity = [
    1,
    0,
    1,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0
]


data = []

for smiles in smiles_list:

    mol = Chem.MolFromSmiles(smiles)

    MW = Descriptors.MolWt(mol)
    LogP = Descriptors.MolLogP(mol)
    TPSA = Descriptors.TPSA(mol)
    HBA = Descriptors.NumHAcceptors(mol)
    HBD = Descriptors.NumHDonors(mol)

    data.append([
        smiles,
        MW,
        LogP,
        TPSA,
        HBA,
        HBD
    ])


df = pd.DataFrame(
    data,
    columns=[
        "SMILES",
        "MW",
        "LogP",
        "TPSA",
        "HBA",
        "HBD"
    ]
)

df["Active"] = activity

print("\n==============================")
print("DATAFRAME")
print("==============================")

print(df)


X = df.drop(["SMILES", "Active"], axis=1)

y = df["Active"]

print("\nFeature Matrix Shape")

print(X.shape)

print("\nTarget Shape")

print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.25,
    random_state=42

)

print("\nTraining Samples")

print(len(X_train))

print("\nTesting Samples")

print(len(X_test))


model = RandomForestClassifier(

    n_estimators=100,
    random_state=42

)


model.fit(X_train, y_train)

print("\nModel Training Completed!")


predictions = model.predict(X_test)

print("\nPredicted Labels")

print(predictions)

print("\nActual Labels")

print(y_test.values)


accuracy = accuracy_score(

    y_test,
    predictions

)

print("\nAccuracy")

print(accuracy)


probabilities = model.predict_proba(X_test)

print("\nPrediction Probabilities")

print(probabilities)


print("\nClasses")

print(model.classes_)


importance = model.feature_importances_

print("\nFeature Importance")

feature_names = X.columns

for feature, score in zip(feature_names, importance):

    print(f"{feature:<10} : {score:.3f}")


new_molecule = pd.DataFrame({

    "MW":[300],
    "LogP":[2.5],
    "TPSA":[55],
    "HBA":[3],
    "HBD":[1]

})

prediction = model.predict(new_molecule)

probability = model.predict_proba(new_molecule)

print("\nPrediction For New Molecule")

print(prediction)

print("\nProbability")

print(probability)
