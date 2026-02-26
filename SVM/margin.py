from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Generate dataset
# X, y = make_blobs(n_samples=200, centers=2, random_state=42)
X, y = make_blobs(n_samples=200, centers=2, cluster_std=3, random_state=42)

# Train linear SVM
model = SVC(kernel='linear', C=1e6)  # large C for hard margin
model.fit(X, y)

# Get parameters
w = model.coef_[0]
b = model.intercept_[0]

# Create grid for plotting
x_min, x_max = X[:, 0].min(), X[:, 0].max()
x_points = np.linspace(x_min, x_max, 100)

# Decision boundary
y_decision = -(w[0] * x_points + b) / w[1]

# Margin lines
margin = 1 / np.linalg.norm(w)
y_margin_pos = y_decision + margin
y_margin_neg = y_decision - margin

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')

plt.plot(x_points, y_decision, 'k-', label='Decision Boundary')
plt.plot(x_points, y_margin_pos, 'k--', label='Margin')
plt.plot(x_points, y_margin_neg, 'k--')

# Highlight support vectors
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150,
            facecolors='none',
            edgecolors='black',
            label='Support Vectors')

plt.title("SVM with Maximum Margin and Support Vectors")
plt.legend()
plt.show()

print("Margin width =", 2 / np.linalg.norm(w))