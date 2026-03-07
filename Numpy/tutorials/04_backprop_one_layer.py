import numpy as np


# Sigmoid activation.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# If a = sigmoid(z), then d(sigmoid)/dz = a*(1-a).
def sigmoid_derivative_from_activation(a):
    return a * (1.0 - a)


# Input batch (N=3, D_in=2).
X = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

# Binary targets (N=3, D_out=1).
y = np.array([[1.0], [1.0], [0.0]])

# Weights and bias.
W = np.array([[0.2], [-0.1]])
b = np.array([[0.0]])

# Forward: Z = XW + b, y_pred = sigmoid(Z).
Z = X @ W + b
y_pred = sigmoid(Z)

# MSE: mean((y - y_pred)^2).
loss = np.mean((y - y_pred) ** 2)

# Backward math (MSE + sigmoid output):
# dA = 2*(y_pred - y)/N
N = X.shape[0]
dA = (2.0 / N) * (y_pred - y)

# dZ = dA * sigmoid'(Z), but using y_pred form: y_pred*(1-y_pred)
dZ = dA * sigmoid_derivative_from_activation(y_pred)

# dW = X^T @ dZ
dW = X.T @ dZ

# db = sum(dZ across batch)
db = np.sum(dZ, axis=0, keepdims=True)

# Gradient descent update.
lr = 0.5
new_W = W - lr * dW
new_b = b - lr * db

print("loss =", loss)
print("dW =\n", dW)
print("db =\n", db)
print("new_W =\n", new_W)
print("new_b =\n", new_b)

print("\nFunction Focus:")
print("- X @ W: forward linear transform")
print("- np.mean: loss")
print("- elementwise *: chain rule combine gradients")
print("- X.T @ dZ: weight gradient")
print("- np.sum(axis=0, keepdims=True): bias gradient")
