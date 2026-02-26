from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D linearly separable data
X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=42)

# Train linear SVM
model = SVC(kernel='linear', C=1e6)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]
model.support_vectors_
model.dual_coef_
print("Number of support vectors:", len(model.support_vectors_))

# Create grid for plane
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 20),
    np.linspace(X[:,1].min(), X[:,1].max(), 20)
)

# Decision plane: w1x + w2y + w3z + b = 0
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

# Margin planes: w^T x + b = ±1
zz_margin_pos = -(w[0]*xx + w[1]*yy + b - 1) / w[2]
zz_margin_neg = -(w[0]*xx + w[1]*yy + b + 1) / w[2]

# Plot
fig = plt.figure(figsize=(9,7))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='coolwarm')

# Highlight support vectors
ax.scatter(model.support_vectors_[:,0],
           model.support_vectors_[:,1],
           model.support_vectors_[:,2],
           s=200,
           facecolors='none',
           edgecolors='black',
           linewidth=2,
           label='Support Vectors')

# Plot decision plane
ax.plot_surface(xx, yy, zz, alpha=0.3)

# Plot margin planes
ax.plot_surface(xx, yy, zz_margin_pos, alpha=0.15)
ax.plot_surface(xx, yy, zz_margin_neg, alpha=0.15)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

plt.title("3D Linear SVM with Margin Planes and Support Vectors")
plt.legend()
plt.show()

# Print margin width
margin = 2 / np.linalg.norm(w)
print("Margin width =", margin)