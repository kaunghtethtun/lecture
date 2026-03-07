import numpy as np
np.random.seed(42)

X = np.random.randn(4,3)
print("Shape of X:", X.shape)
W = np.random.randn(3,2)
print("Shape of W:", W.shape)
print("W:\n", W)
b=np.random.randn(2,)
print("Shape of b:", b.shape)
print("b:", b)
print("transpose of b:", b.T)
Z = X @ W + b
print("Shape of Z:", Z.shape)
print("Z:\n", Z)