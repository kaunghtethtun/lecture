import torch

x = torch.rand(3, requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
print(z)

z.backward(y)
print(x.grad)

#x.requires_grad_(False)
#x.detach()
#with torch.no_grad():