import torch

# 1. Define inputs. 'requires_grad=True' tells PyTorch to track math for this variable.
x = torch.tensor(1.5)
w = torch.tensor(0.8, requires_grad=True)
b = torch.tensor(0.1)
target = torch.tensor(2.0)

# 2. Forward Pass (Just write the equation)
z = (w * x) + b
a = z
loss = (a - target)**2

# 3. Backward Pass (The Magic Step)
loss.backward()

# 4. Check the gradient
print(f"PyTorch Gradient (w.grad): {w.grad:.2f}")