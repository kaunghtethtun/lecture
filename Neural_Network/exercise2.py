import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# Exercise 2:
# Sigmoid derivative and chain-rule gradient
# ------------------------------------------------------------

# Given values
# a: output from sigmoid neuron
# dL_da: incoming gradient from next layer
a = np.array(0.8)
dL_da = np.array(0.5)

# Step 1: Local derivative of sigmoid
# If a = sigmoid(z), then da/dz = a * (1 - a)
one_minus_a = 1 - a
da_dz = a * one_minus_a

# Step 2: Chain rule to pass gradient backward
# dL/dz = dL/da * da/dz
dL_dz = dL_da * da_dz

# Step-by-step output for learning
print("Exercise 2: Sigmoid Local Gradient + Chain Rule")
print(f"Given: a = {a:.4f}, dL/da = {dL_da:.4f}")
print(f"Step 1: (1 - a) = 1 - {a:.4f} = {one_minus_a:.4f}")
print(f"Step 2: da/dz = a * (1 - a) = {a:.4f} * {one_minus_a:.4f} = {da_dz:.4f}")
print(f"Step 3: dL/dz = dL/da * da/dz = {dL_da:.4f} * {da_dz:.4f} = {dL_dz:.4f}")
print(f"Final: local gradient da/dz = {da_dz:.4f}")
print(f"Final: backprop gradient dL/dz = {dL_dz:.4f}")

# ----------------- Visualization -----------------
# Left plot: sigmoid local derivative curve as a function of a
a_values = np.linspace(0.0, 1.0, 400)
derivative_values = a_values * (1 - a_values)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle("Exercise 2: Sigmoid Gradient Visualization", fontsize=13)

axes[0].plot(a_values, derivative_values, color="royalblue", label="da/dz = a(1-a)")
axes[0].scatter([a], [da_dz], color="crimson", zorder=3, label=f"at a={a:.1f}")
axes[0].set_xlabel("Sigmoid Output a")
axes[0].set_ylabel("Local Gradient da/dz")
axes[0].set_title("Sigmoid Local Gradient Curve")
axes[0].grid(alpha=0.3)
axes[0].legend()

# Right plot: chain-rule components for this example
bars = ["dL/da", "da/dz", "dL/dz"]
values = [dL_da, da_dz, dL_dz]
axes[1].bar(bars, values, color=["orange", "teal", "purple"])
axes[1].set_title("Chain Rule Components")
axes[1].set_ylabel("Value")
axes[1].grid(axis="y", alpha=0.3)

for idx, value in enumerate(values):
    axes[1].text(idx, value + (0.02 if value >= 0 else -0.04), f"{value:.3f}", ha="center")

plt.tight_layout()
script_dir = Path(__file__).resolve().parent
output_path = script_dir / "exercise2_visual.png"

if "agg" in plt.get_backend().lower():
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to: {output_path}")
else:
    plt.show()
