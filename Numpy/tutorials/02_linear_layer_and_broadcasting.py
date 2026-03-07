import numpy as np

np.random.seed(42)  

# np.array: batch input, shape (N, D_in).
X = np.array(
    [
        [1.0, 2.0],
        [0.5, -1.0],
        [0.0, 1.0],
    ]
)

# np.random.randn: initialize weights with normal distribution.
W = np.random.randn(2, 3) * 0.1

# np.zeros: initialize biases as zeros.
b = np.zeros((1, 3))

# @ operator: matrix multiplication.
# broadcasting: b (1, 3) is added to each row of X @ W (3, 3).
Z = X @ W + b

print("X.shape ->", X.shape)
print("W.shape ->", W.shape)
print("b.shape ->", b.shape)
print("Z.shape ->", Z.shape)
print("Z =\n", Z)

# np.sum with axis=0 and keepdims=True: common for bias gradient shape.
column_sum = np.sum(Z, axis=0, keepdims=True)
print("np.sum(Z, axis=0, keepdims=True).shape ->", column_sum.shape)
print("column_sum =\n", column_sum)
print("\nFunction Focus:")
print("- np.random.seed: deterministic examples")
print("- np.random.randn: random weight init")
print("- np.zeros: bias init")
print("- @: matrix multiply in linear layers")
print("- broadcasting (+ b): add bias to every sample")
print("- np.sum(axis=0, keepdims=True): keep bias-compatible shape")
