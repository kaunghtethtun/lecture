import numpy as np

# ------------------------------------------------------------
# This script demonstrates:
# 1) Forward propagation
# 2) Loss calculation (MSE)
# 3) Backpropagation (manual gradients)
# 4) Weight update using different learning rates
#
# Network shape: 1 input -> 1 hidden(tanh) -> 1 output(linear)
# ------------------------------------------------------------

# Input feature value for one training sample
x = 0.5

# Ground-truth target for this sample
y_true = 0.8

# Initial model weights:
# w1: input -> hidden
# w2: hidden -> output
initial_w1, initial_w2 = 0.4, 0.7

# Learning rates to compare (small, medium, large)
learning_rates = [0.01, 0.1, 1.0]


def one_step_training(x, y_true, w1, w2, lr):
    # ----------------- Forward Propagation -----------------
    # Hidden pre-activation
    hidden_z = x * w1
    # Hidden activation (nonlinear)
    hidden_a = np.tanh(hidden_z)
    # Output pre-activation
    output_z = hidden_a * w2
    # Predicted output (linear output neuron)
    y_pred = output_z

    # ----------------- Loss Computation -----------------
    # MSE for one sample: 0.5 * (target - prediction)^2
    loss = 0.5 * (y_true - y_pred) ** 2

    # ----------------- Backpropagation -----------------
    # dLoss/dy_pred for 0.5*(y_true - y_pred)^2
    error = -(y_true - y_pred)
    # Gradient for w2: dLoss/dw2 = dLoss/dy_pred * dy_pred/dw2 = error * hidden_a
    d_w2 = error * hidden_a
    # Derivative of tanh(hidden_z)
    d_tanh = 1 - np.square(hidden_a)
    # Gradient for w1: dLoss/dw1 = error * w2 * d_tanh * x
    d_w1 = error * w2 * d_tanh * x

    # ----------------- Gradient Descent Update -----------------
    new_w1 = w1 - lr * d_w1
    new_w2 = w2 - lr * d_w2

    # ----------------- Forward Again (after update) -----------------
    new_hidden_a = np.tanh(x * new_w1)
    new_y_pred = new_hidden_a * new_w2
    new_loss = 0.5 * (y_true - new_y_pred) ** 2

    return loss, new_loss, d_w1, d_w2, new_w1, new_w2


print("Manual NN training (single sample, one update step)")
print(f"Initial weights: w1={initial_w1:.4f}, w2={initial_w2:.4f}")
print(f"Input x={x}, Target y={y_true}")

for lr in learning_rates:
    loss, new_loss, d_w1, d_w2, new_w1, new_w2 = one_step_training(
        x=x,
        y_true=y_true,
        w1=initial_w1,
        w2=initial_w2,
        lr=lr,
    )

    print(f"\nLearning Rate = {lr}")
    print(f"Initial Loss: {loss:.6f}")
    print(f"Gradients: d_w1={d_w1:.6f}, d_w2={d_w2:.6f}")
    print(f"Updated weights: w1={new_w1:.6f}, w2={new_w2:.6f}")
    print(f"Loss after 1 update: {new_loss:.6f}")
