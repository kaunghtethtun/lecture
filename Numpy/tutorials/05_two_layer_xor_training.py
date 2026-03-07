import numpy as np

np.random.seed(42)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)


# XOR dataset.
X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)
y = np.array([[0.0], [1.0], [1.0], [0.0]])

# Dimensions.
input_dim = 2
hidden_dim = 4
output_dim = 1

# np.random.randn + np.zeros: init parameters.
W1 = np.random.randn(input_dim, hidden_dim) * 0.5
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.5
b2 = np.zeros((1, output_dim))

lr = 0.8
epochs = 5000
N = X.shape[0]

for epoch in range(1, epochs + 1):
    # Forward pass.
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    y_pred = sigmoid(Z2)

    # Loss.
    loss = np.mean((y - y_pred) ** 2)

    # Backward pass.
    dA2 = (2.0 / N) * (y_pred - y)
    dZ2 = dA2 * sigmoid_derivative(y_pred)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update.
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss {loss:.6f}")

print("\nFinal outputs:")
print(y_pred)
print("Binary predictions:")
print((y_pred > 0.5).astype(int))
print("Targets:")
print(y.astype(int))

print("\nFunction Focus:")
print("- Full forward/backward pipeline with matrix operations")
print("- Most used NumPy ops: @, np.mean, np.sum, .T, np.exp, broadcasting")
