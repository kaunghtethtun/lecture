import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Exercise 3:
# Vanishing-gradient style example across 4 sigmoid layers
# ------------------------------------------------------------

# Given values
# num_layers: number of chained local gradients
# local_grad: each sigmoid local derivative value
# learning_rate: gradient descent step size
num_layers = 4
local_grad = 0.1
learning_rate = 0.01

# Step 1: Total gradient at the first layer
# Product of all local gradients:
# (0.1)^4 = 0.1 * 0.1 * 0.1 * 0.1
total_grad = np.power(local_grad, num_layers)

# Step 2: Weight update magnitude for gradient descent
# |delta_w| = learning_rate * |gradient|
# Here we assume dL/dw = total_grad
weight_change = learning_rate * total_grad

# Step-by-step output for learning
print("Exercise 3: Gradient Shrinkage Across Layers")
print(f"Given: num_layers = {num_layers}, local_grad = {local_grad}, learning_rate = {learning_rate}")

# Show explicit multiplication chain
chain_terms = " * ".join([f"{local_grad:.1f}" for _ in range(num_layers)])
print(f"Step 1: total_grad = {chain_terms} = {total_grad:.8f}")
print(f"Step 2: weight_change = learning_rate * total_grad = {learning_rate:.4f} * {total_grad:.8f} = {weight_change:.10f}")

# Step 3: Discussion-style interpretation
if weight_change < 1e-4:
    print("Discussion: This update is extremely small.")
    print("Result: The first-layer weight changes very little each step.")
    print("Conclusion: Learning is likely too slow (vanishing gradient behavior).")
else:
    print("Discussion: This update size may be usable for learning.")

# ----------------- Visualization -----------------
# Layer depth counted from output side:
# k=1 means closest layer to output, k=num_layers means earliest layer.
k_values = np.arange(1, num_layers + 1)
gradient_by_depth = np.power(local_grad, k_values)
update_by_depth = learning_rate * gradient_by_depth

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Exercise 3: Vanishing Gradient Visualization", fontsize=13)

# Left: gradient magnitude decay by depth
axes[0].plot(k_values, gradient_by_depth, marker="o", color="crimson")
axes[0].set_title("Gradient Magnitude by Layer Depth")
axes[0].set_xlabel("Depth k (from output)")
axes[0].set_ylabel("|Gradient| = local_grad^k")
axes[0].set_xticks(k_values)
axes[0].set_yscale("log")
axes[0].grid(alpha=0.3, which="both")

# Right: corresponding weight-update magnitudes
axes[1].bar(k_values, update_by_depth, color="steelblue")
axes[1].set_title("Weight Update Magnitude by Depth")
axes[1].set_xlabel("Depth k (from output)")
axes[1].set_ylabel("|delta_w| = lr * local_grad^k")
axes[1].set_xticks(k_values)
axes[1].set_yscale("log")
axes[1].grid(axis="y", alpha=0.3, which="both")

for depth, upd in zip(k_values, update_by_depth):
    axes[1].text(depth, upd * 1.3, f"{upd:.1e}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
script_dir = Path(__file__).resolve().parent
output_path = script_dir / "exercise3_visual.png"

if "agg" in plt.get_backend().lower():
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to: {output_path}")
else:
    plt.show()
