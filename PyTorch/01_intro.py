import torch

x = torch.ones(2,3,2, dtype=torch.int32)
print(x.dtype)
print(x.size())

y = torch.tensor([1,2])
print(y)

a = torch.rand(2,3)
b = torch.rand(2,3)

print(a)
# print(b)
# print(a+b)
# print(torch.add(a,b))
# print(torch.sub(a,b))
# print(torch.mul(a,b))
# print(torch.div(a,b))
# print(torch.matmul(a,b.T))

print(a[:,0])
print(a[1,1].item())

e = torch.rand(4,4)
print(e)
f = e.view(2,8)
print(f)
g = e.view(4,-1)
print(g)
print(g.reshape(2,8))