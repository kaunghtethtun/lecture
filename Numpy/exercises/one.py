import numpy as np
X = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
])
print("Shape of X:", X.shape)
print("Transpose of X:\n", X.T)
print("Column means:", X.mean(axis=0))
print("Row means:", X.mean(axis=1))