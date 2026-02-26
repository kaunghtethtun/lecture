import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# -----------------------------
# 1️⃣ Generate Synthetic Data
# -----------------------------
X, y = make_blobs(n_samples=300,
                  centers=2,
                  cluster_std=2.0,
                  random_state=42)

# -----------------------------
# 2️⃣ Train Models
# -----------------------------
nb = GaussianNB()
nb.fit(X, y)

svm = SVC(kernel='linear')
svm.fit(X, y)

# -----------------------------
# 3️⃣ Create Mesh Grid
# -----------------------------
x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]

# -----------------------------
# 4️⃣ Predictions
# -----------------------------
Z_nb = nb.predict(grid)
Z_nb = Z_nb.reshape(xx.shape)

Z_svm = svm.predict(grid)
Z_svm = Z_svm.reshape(xx.shape)

# -----------------------------
# 5️⃣ Plot Naive Bayes
# -----------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_nb, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')
plt.title("Gaussian Naive Bayes")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# -----------------------------
# 6️⃣ Plot SVM
# -----------------------------
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_svm, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

# Plot SVM margin
w = svm.coef_[0]
b = svm.intercept_[0]

x_line = np.linspace(x_min, x_max, 100)
y_line = -(w[0] * x_line + b) / w[1]
plt.plot(x_line, y_line)

plt.title("Linear SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()