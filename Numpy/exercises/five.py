import numpy as np

np.random.seed(42)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.random.randn(4, 3)
print("X shape:", X.shape)
print("X:\n", X)

W1 = np.random.randn(3, 4)
b1 = np.random.randn(1, 4)

W2 = np.random.randn(4, 1)
b2 = np.random.randn(1, 1)

Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

print("\nW1 shape:", W1.shape)
print("b1 shape:", b1.shape)
print("Z1 shape:", Z1.shape)
print("A1 shape:", A1.shape)

print("\nW2 shape:", W2.shape)
print("b2 shape:", b2.shape)
print("Z2 shape:", Z2.shape)
print("A2 shape:", A2.shape)

print("\nA2 (final output):\n", A2)
