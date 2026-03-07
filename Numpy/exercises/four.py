import numpy as np
def relu(x):
    return np.maximum(0, x)
x = np.array([-3,-1,0,2,5])
print("x:", x)
print("relu(x):", relu(x))