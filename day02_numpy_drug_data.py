import numpy as np
#1-D Arrays
ic50 = np.array([12,45,8,20])
tox=np.array([2,5,1])
#vectorisaion , mechanism used is Broadcasting
print(ic50 * 2)
#Determining the number of rows and columns
print(ic50.shape)
#indexing 
print(ic50[1])
#slicing
print(ic50[1:4])
#Arrays interacts with Arrays
#print(ic50+tox)

#2D Arrays
expr = np.array([
    [10,20,30],
    [40,50,60]
])

print(expr.shape)
 #Reshaping
x = np.array([
    1,2,3,4,5,6
])
print(x.reshape(2,3))

#Boolean Masking
import numpy as np

ic50 = np.array([10,45,8,20])
print(ic50<20)
print(ic50[ic50 < 20])
#feature matrix and target vector 
X = np.array([
    [300,2.1,1],
    [450,4.5,0],
    [250,1.8,2]
])

y = np.array([
    1,
    0,
    1
])
print(y.shape)#answers
print(x.shape)#features
