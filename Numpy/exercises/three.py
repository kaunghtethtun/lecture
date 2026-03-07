import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
z=np.array([-2,-1,0,1,2])
print("z:", z)
print("sigmoid(z):", sigmoid(z))