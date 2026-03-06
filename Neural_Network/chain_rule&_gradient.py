import numpy as np

# ------------------------------------------------------------
# Goal:
# Show step-by-step chain-rule gradient calculation for a tiny model:
#   z = w*x + b
#   a = z                     (identity activation)
#   L = (a - target)^2
# We want dL/dw.
# ------------------------------------------------------------

# Given values (single training example)
x = 1.5        # input feature
w = 0.8        # weight parameter
b = 0.1        # bias parameter
target = 2.0   # ground-truth output

print("Step-by-step Chain Rule and Gradient Calculation")
print(f"Input values: x={x}, w={w}, b={b}, target={target}")

# ----------------- Forward Pass -----------------
# Step 1: Linear combination
z = (w * x) + b
print(f"\nForward Step 1: z = w*x + b = ({w}*{x}) + {b} = {z:.4f}")

# Step 2: Activation (identity), so output is same as z
a = z
print(f"Forward Step 2: a = z (identity activation) = {a:.4f}")

# Step 3: Squared-error loss
loss = (a - target) ** 2
print(f"Forward Step 3: L = (a - target)^2 = ({a:.4f} - {target})^2 = {loss:.6f}")

# ----------------- Backward Pass -----------------
# Local derivative 1: dL/da = 2*(a - target)
dL_da = 2 * (a - target)
print(f"\nBackward Step 1: dL/da = 2*(a - target) = 2*({a:.4f} - {target}) = {dL_da:.4f}")

# Local derivative 2: da/dz = 1 (because a = z)
da_dz = 1.0
print(f"Backward Step 2: da/dz = 1 (identity activation)")

# Local derivative 3: dz/dw = x (because z = w*x + b)
dz_dw = x
print(f"Backward Step 3: dz/dw = x = {dz_dw:.4f}")

# Chain rule: dL/dw = dL/da * da/dz * dz/dw
gradient_w = dL_da * da_dz * dz_dw
print(
    "Backward Step 4: dL/dw = dL/da * da/dz * dz/dw "
    f"= ({dL_da:.4f}) * ({da_dz:.1f}) * ({dz_dw:.4f}) = {gradient_w:.4f}"
)

# Optional interpretation of sign
if gradient_w > 0:
    direction_note = "positive gradient -> gradient descent will decrease w"
elif gradient_w < 0:
    direction_note = "negative gradient -> gradient descent will increase w"
else:
    direction_note = "zero gradient -> w is locally optimal for this sample"

print(f"\nFinal Gradient (dL/dw): {gradient_w:.4f}")
print(f"Interpretation: {direction_note}")
