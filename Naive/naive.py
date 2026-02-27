import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# 1️⃣ Create dataset
# -----------------------------
np.random.seed(42)

# Class 0 (purple): tight cluster
X0 = np.random.randn(100,2) + [0,10]
y0 = np.zeros(100)

# Class 1 (yellow): wide cluster
X1 = np.random.randn(200,2)*3 + [5,0]
y1 = np.ones(200)

# Combine
X = np.vstack((X0,X1))
y = np.hstack((y0,y1))

# Optional: add outlier
X = np.vstack((X, [[8,15]]))
y = np.hstack((y, [1]))  

# -----------------------------
# 2️⃣ Train Naive Bayes
# -----------------------------
nb = GaussianNB()
nb.fit(X, y)

# -----------------------------
# 3️⃣ Create meshgrid for visualization
# -----------------------------
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

xx, yy = np.meshgrid(np.linspace(x_min,x_max,300),
                     np.linspace(y_min,y_max,300))

grid = np.c_[xx.ravel(), yy.ravel()]

# Predict class for each point in the grid
Z = nb.predict(grid).reshape(xx.shape)

# -----------------------------
# 4️⃣ Plot Naive Bayes prediction
# -----------------------------
plt.figure(figsize=(7,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='Paired')
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap='Paired')
plt.title("Gaussian Naive Bayes Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()