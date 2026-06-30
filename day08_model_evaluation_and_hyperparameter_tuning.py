from rdkit import Chem
from rdkit.Chem import Descriptors

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


smiles = [

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

for s in smiles:

    mol = Chem.MolFromSmiles(s)

    data.append({

        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol)

    })

df = pd.DataFrame(data)

df["Active"] = activity

print(df)


X = df.drop("Active", axis=1)

y = df["Active"]


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.25,
    random_state=42

)


model = RandomForestClassifier(

    n_estimators=100,
    random_state=42

)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)


accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")

print(accuracy)


cm = confusion_matrix(

    y_test,
    predictions

)

print("\nConfusion Matrix")

print(cm)


precision = precision_score(

    y_test,
    predictions

)

print("\nPrecision")

print(precision)


recall = recall_score(

    y_test,
    predictions

)

print("\nRecall")

print(recall)


f1 = f1_score(

    y_test,
    predictions

)

print("\nF1 Score")

print(f1)


roc_auc = roc_auc_score(

    y_test,
    probabilities[:, 1]

)

print("\nROC-AUC")

print(roc_auc)


train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(

    y_train,
    train_predictions

)

print("\nTraining Accuracy")

print(train_accuracy)


test_accuracy = accuracy_score(

    y_test,
    predictions

)

print("\nTesting Accuracy")

print(test_accuracy)


scores = cross_val_score(

    model,
    X,
    y,
    cv=5

)

print("\nCross Validation Scores")

print(scores)

print("\nAverage CV Accuracy")

print(scores.mean())


param_grid = {

    "n_estimators":[50,100,200],

    "max_depth":[5,10,None]

}

grid = GridSearchCV(

    estimator=RandomForestClassifier(
        random_state=42
    ),

    param_grid=param_grid,

    cv=5,

    scoring="accuracy"

)

grid.fit(

    X_train,
    y_train

)

print("\nBest Parameters")

print(grid.best_params_)

print("\nBest Cross Validation Score")

print(grid.best_score_)


best_model = grid.best_estimator_

best_predictions = best_model.predict(X_test)

best_accuracy = accuracy_score(

    y_test,
    best_predictions

)

print("\nBest Model Test Accuracy")

print(best_accuracy)
