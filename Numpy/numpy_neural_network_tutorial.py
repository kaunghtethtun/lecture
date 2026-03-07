import numpy as np


# Make results reproducible.
np.random.seed(42)


# ------------------------------------------------------------
# 1) NumPy basics for neural networks
# ------------------------------------------------------------
print("\n[1] NumPy basics")
X_demo = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)
print("Input matrix X:\n", X_demo)
print("Shape of X:", X_demo.shape)  # (batch_size, features)


# ------------------------------------------------------------
# 2) Core NN operations: linear layer + activations
# ------------------------------------------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    """Derivative of sigmoid with respect to pre-activation.
    Here `a` is sigmoid(z), so derivative is a*(1-a).
    """
    return a * (1.0 - a)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


print("\n[2] Forward operations")
W_demo = np.random.randn(2, 3) * 0.1
b_demo = np.zeros((1, 3))

Z_demo = X_demo @ W_demo + b_demo
A_demo = sigmoid(Z_demo)

print("W shape:", W_demo.shape)
print("b shape:", b_demo.shape)
print("Z shape:", Z_demo.shape)
print("A shape:", A_demo.shape)


# ------------------------------------------------------------
# 3) Build a tiny 2-layer network (XOR task)
#    Input(2) -> Hidden(4, sigmoid) -> Output(1, sigmoid)
# ------------------------------------------------------------
print("\n[3] Training a tiny neural network on XOR")

X = X_demo
y = np.array([[0.0], [1.0], [1.0], [0.0]])

input_dim = 2
hidden_dim = 4
output_dim = 1

W1 = np.random.randn(input_dim, hidden_dim) * 0.5
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.5
b2 = np.zeros((1, output_dim))

learning_rate = 0.8
epochs = 5000
n = X.shape[0]

for epoch in range(1, epochs + 1):
    # Forward pass
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)

    Z2 = A1 @ W2 + b2
    y_pred = sigmoid(Z2)

    loss = mse_loss(y, y_pred)

    # Backward pass (MSE + sigmoid output)
    # dLoss/dy_pred = 2*(y_pred-y)/n
    dA2 = (2.0 / n) * (y_pred - y)
    dZ2 = dA2 * sigmoid_derivative(y_pred)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Gradient descent update
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")


# ------------------------------------------------------------
# 4) Evaluate results
# ------------------------------------------------------------
print("\n[4] Final predictions")
print("Raw outputs:\n", y_pred)
print("Binary predictions:\n", (y_pred > 0.5).astype(int))
print("Targets:\n", y.astype(int))

print("\nTutorial complete.")
