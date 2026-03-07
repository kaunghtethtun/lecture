import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    return np.maximum(0.0, z)

def tanh(z):
    return np.tanh(z)

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_prob):
    eps = 1e-8
    p = np.clip(y_prob, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))


Z = np.array([[-2.0, 0.0, 2.0]])
y_true = np.array([[0.0, 1.0, 1.0]])
y_prob = sigmoid(Z)

print("Z =", Z)
print("sigmoid(Z) =", y_prob)
print("relu(Z) =", relu(Z))
print("tanh(Z) =", tanh(Z))
print("MSE(y_true, y_prob) =", mse_loss(y_true, y_prob))
print("BCE(y_true, y_prob) =", binary_cross_entropy(y_true, y_prob))

print("\nFunction Focus:")
print("- np.exp: sigmoid/softmax math")
print("- np.maximum: ReLU")
print("- np.tanh: tanh activation")
print("- np.mean: average loss")
print("- np.clip: avoid log(0)")
print("- np.log: cross-entropy")
