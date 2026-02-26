from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Generate data
X, y = make_blobs(n_samples=200, centers=2, random_state=42)

model = SVC(kernel='linear', C=1e6)
model.fit(X, y)

w_original = model.coef_[0]
b_original = model.intercept_[0]

print("Original # Support Vectors:", len(model.support_vectors_))

# Find non-support vector indices
all_indices = np.arange(len(X))
sv_indices = model.support_
# non_sv_indices = np.setdiff1d(all_indices, sv_indices)

# # Pick one far-away non-support vector
# remove_index = non_sv_indices[0]

# X_new = np.delete(X, remove_index, axis=0)
# y_new = np.delete(y, remove_index)

# # Retrain
# model_new = SVC(kernel='linear', C=1e6)
# model_new.fit(X_new, y_new)

# w_new = model_new.coef_[0]
# b_new = model_new.intercept_[0]

# print("Did boundary change?",
#       np.allclose(w_original, w_new))

# Remove first support vector
remove_index = sv_indices[0]

X_new2 = np.delete(X, remove_index, axis=0)
y_new2 = np.delete(y, remove_index)

# Retrain
model_new2 = SVC(kernel='linear', C=1e6)
model_new2.fit(X_new2, y_new2)

w_new2 = model_new2.coef_[0]
b_new2 = model_new2.intercept_[0]

print("Did boundary change?",
      np.allclose(w_original, w_new2))

plt.title("3D Linear SVM with Margin Planes and Support Vectors")
plt.legend()
plt.show()