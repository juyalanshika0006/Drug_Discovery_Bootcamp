#import libraries
import numpy as np
#import test train split
from sklearn.model_selection import train_test_split
#import machine learning model
from sklearn.ensemble import RandomForestClassifier
# import accuracy calculator
from sklearn.metrics import accuracy_score
#creating Features
X = np.array([
    [300, 2.1],   # Compound 1
    [450, 4.5],   # Compound 2
    [250, 1.8],   # Compound 3
    [500, 5.0],   # Compound 4
    [350, 3.0],   # Compound 5
    [280, 2.0],   # Compound 6
    [470, 4.8],   # Compound 7
    [260, 1.5]    # Compound 8
])

# Columns:
# Molecular Weight (MW)
# LogP

#creating labels
y = np.array([
    1,  # Active
    0,  # Inactive
    1,
    0,
    1,
    1,
    0,
    1
])
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

