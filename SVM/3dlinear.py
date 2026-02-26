from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate 3D data
X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=42)

from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='coolwarm')

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

plt.title("3D Linearly Separable Data")
plt.show()

# Create grid
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min(), X[:,0].max(), 20),
    np.linspace(X[:,1].min(), X[:,1].max(), 20)
)

# Compute z from plane equation
zz = -(w[0]*xx + w[1]*yy + b) / w[2]

# Plot everything
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], X[:,2], c=y, cmap='coolwarm')
ax.plot_surface(xx, yy, zz, alpha=0.3)

ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")

plt.title("Linear SVM Plane in 3D")
plt.show()