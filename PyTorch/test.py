import torch
import time

device = "cuda"

a = torch.rand(3000,3000, device=device)
b = torch.rand(3000,3000, device=device)

start = time.time()
c = a @ b
torch.cuda.synchronize()
print("Time:", time.time()-start)