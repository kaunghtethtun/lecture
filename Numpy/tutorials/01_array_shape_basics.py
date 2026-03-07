import numpy as np

X = np.array(
    [
        [0.0, 3.0],
        [4.0, 0.0],
        [1.0, 1.0],
    ]
)

y = np.array([[1.0], [1.0], [0.0]])

print("X =\n", X)
print("y =\n", y)
print("X.shape ->", X.shape)  
print("y.shape ->", y.shape) 

v = np.array([0.2, 0.4, 0.6])
v_col = v.reshape(3, 1)
print("v_col ", v_col)
print("v.shape ->", v.shape)
print("v_col.shape ->", v_col.shape)

print("X.T shape ->", X.T.shape)

one_sample = X[0] 
one_sample_batch = np.expand_dims(one_sample, axis=0)  
print("one_sample ", one_sample)
print("one_sample_batch ", one_sample_batch)
print("one_sample.shape ->", one_sample.shape)
print("one_sample_batch.shape ->", one_sample_batch.shape)

back_to_vector = np.squeeze(v_col)
print("back_to_vector.shape ->", back_to_vector.shape)

print("\nFunction Focus:")
print("- np.array: build tensors")
print("- .shape: inspect dimensions")
print("- .reshape: change tensor shape")
print("- .T: transpose for linear algebra")
print("- np.expand_dims / np.squeeze: add or remove singleton dimensions")
