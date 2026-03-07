"""Lecture-aligned NumPy exercises for neural-network fundamentals.

How to use:
1) Read each exercise prompt.
2) Fill in your answers in the TODO sections.
3) Run: python3 Numpy/lecture_exercises.py

The script auto-checks your answers and prints a score.
"""

import numpy as np

np.random.seed(42)


# ============================================================
# Exercise 1: Sigmoid derivative + chain rule
# From your lecture: da/dz = a(1-a), dL/dz = dL/da * da/dz
# ============================================================
print("\nExercise 1: Sigmoid derivative + chain rule")
a = 0.8
dL_da = 0.5

# TODO: Replace these with your computed values
ex1_da_dz = 0.16
ex1_dL_dz = 0.08

expected_da_dz = a * (1 - a)
expected_dL_dz = dL_da * expected_da_dz

ex1_pass = np.isclose(ex1_da_dz, expected_da_dz) and np.isclose(ex1_dL_dz, expected_dL_dz)
print(f"Your da/dz={ex1_da_dz:.6f}, expected={expected_da_dz:.6f}")
print(f"Your dL/dz={ex1_dL_dz:.6f}, expected={expected_dL_dz:.6f}")
print("Result:", "PASS" if ex1_pass else "FAIL")


# ============================================================
# Exercise 2: Forward pass (matrix shapes)
# Z = XW + b, A = sigmoid(Z)
# ============================================================
print("\nExercise 2: Forward pass shapes and values")
X = np.array([[1.0, 2.0], [0.5, -1.0]])
W = np.array([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]])
b = np.array([[0.1, 0.0, -0.1]])

# TODO: Replace these with your computed answers
ex2_Z = X @ W + b
ex2_A = 1.0 / (1.0 + np.exp(-ex2_Z))

expected_Z = X @ W + b
expected_A = 1.0 / (1.0 + np.exp(-expected_Z))

ex2_pass = ex2_Z.shape == (2, 3) and np.allclose(ex2_Z, expected_Z) and np.allclose(ex2_A, expected_A)
print("Your Z shape:", ex2_Z.shape, "expected:", (2, 3))
print("Result:", "PASS" if ex2_pass else "FAIL")


# ============================================================
# Exercise 3: One-step gradient descent
# Single neuron with linear output: y_pred = x*w
# Loss: 0.5*(y - y_pred)^2
# ============================================================
print("\nExercise 3: One-step gradient update")
x = 2.0
y_true = 1.0
w = 0.3
lr = 0.1

# TODO: Replace with your computed values
ex3_y_pred = x * w
ex3_grad = (ex3_y_pred - y_true) * x
ex3_new_w = w - lr * ex3_grad

expected_y_pred = x * w
expected_grad = (expected_y_pred - y_true) * x
expected_new_w = w - lr * expected_grad

ex3_pass = (
    np.isclose(ex3_y_pred, expected_y_pred)
    and np.isclose(ex3_grad, expected_grad)
    and np.isclose(ex3_new_w, expected_new_w)
)
print(f"Your y_pred={ex3_y_pred:.6f}, expected={expected_y_pred:.6f}")
print(f"Your grad={ex3_grad:.6f}, expected={expected_grad:.6f}")
print(f"Your new_w={ex3_new_w:.6f}, expected={expected_new_w:.6f}")
print("Result:", "PASS" if ex3_pass else "FAIL")


# ============================================================
# Exercise 4: Vanishing gradient style computation
# total_grad = local_grad^num_layers
# update = lr * total_grad
# ============================================================
print("\nExercise 4: Vanishing-gradient calculation")
num_layers = 5
local_grad = 0.1
lr = 0.01

# TODO: Replace with your computed values
ex4_total_grad = local_grad ** num_layers
ex4_update = lr * ex4_total_grad

expected_total_grad = local_grad ** num_layers
expected_update = lr * expected_total_grad

ex4_pass = np.isclose(ex4_total_grad, expected_total_grad) and np.isclose(ex4_update, expected_update)
print(f"Your total_grad={ex4_total_grad:.10f}, expected={expected_total_grad:.10f}")
print(f"Your update={ex4_update:.12f}, expected={expected_update:.12f}")
print("Result:", "PASS" if ex4_pass else "FAIL")


# ============================================================
# Exercise 5: XOR mini-training sanity check
# Check that loss decreases after training steps.
# ============================================================
print("\nExercise 5: XOR training sanity check")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1.0 - a)


X = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)
y = np.array([[0.0], [1.0], [1.0], [0.0]])

W1 = np.random.randn(2, 3) * 0.3
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.3
b2 = np.zeros((1, 1))

n = X.shape[0]
eta = 0.8

# initial loss
A1 = sigmoid(X @ W1 + b1)
y_pred = sigmoid(A1 @ W2 + b2)
initial_loss = np.mean((y - y_pred) ** 2)

for _ in range(2000):
    Z1 = X @ W1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + b2
    y_pred = sigmoid(Z2)

    dA2 = (2.0 / n) * (y_pred - y)
    dZ2 = dA2 * sigmoid_derivative(y_pred)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * sigmoid_derivative(A1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    W2 -= eta * dW2
    b2 -= eta * db2
    W1 -= eta * dW1
    b1 -= eta * db1

final_loss = np.mean((y - y_pred) ** 2)

# TODO: Replace with True if your reasoning says the condition below should hold.
ex5_loss_decreased = final_loss < initial_loss

ex5_pass = bool(ex5_loss_decreased)
print(f"Initial loss={initial_loss:.6f}, Final loss={final_loss:.6f}")
print("Result:", "PASS" if ex5_pass else "FAIL")


# ============================================================
# Score Summary
# ============================================================
results = [ex1_pass, ex2_pass, ex3_pass, ex4_pass, ex5_pass]
score = int(sum(results))
print("\n" + "=" * 54)
print(f"Total Score: {score}/5")
print("=" * 54)

if score == 5:
    print("Excellent. You can now implement a small NN with NumPy.")
elif score >= 3:
    print("Good progress. Recheck failed exercises and shape logic.")
else:
    print("Review the README and tutorial script, then retry.")
