import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -----------------------------
# 1️⃣ Create modified dataset
# -----------------------------
np.random.seed(42)

# Purple: tight cluster
X1 = np.random.randn(100,2) + [0,10]
y1 = np.zeros(100)

# Yellow: wide spread
X2 = np.random.randn(200,2)*3 + [5,0]
y2 = np.ones(200)

# Combine
X = np.vstack((X1,X2))
y = np.hstack((y1,y2))

# Optional: add outlier
X = np.vstack((X, [[8,15]]))
y = np.hstack((y, [1]))  

# -----------------------------
# 2️⃣ Train models
# -----------------------------
nb = GaussianNB()
nb.fit(X,y)

svm = SVC(kernel='linear')
svm.fit(X,y)

# -----------------------------
# 3️⃣ Create mesh grid
# -----------------------------
x_min, x_max = X[:,0].min()-2, X[:,0].max()+2
y_min, y_max = X[:,1].min()-2, X[:,1].max()+2

xx, yy = np.meshgrid(np.linspace(x_min,x_max,300),
                     np.linspace(y_min,y_max,300))

grid = np.c_[xx.ravel(), yy.ravel()]

# -----------------------------
# 4️⃣ Predictions
# -----------------------------
Z_nb = nb.predict(grid).reshape(xx.shape)
Z_svm = svm.predict(grid).reshape(xx.shape)

# -----------------------------
# 5️⃣ Plot
# -----------------------------
plt.figure(figsize=(12,5))

# Naive Bayes
plt.subplot(1,2,1)
plt.contourf(xx,yy,Z_nb,alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k')
plt.title("Gaussian Naive Bayes (Curved boundary)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# Linear SVM
plt.subplot(1,2,2)
plt.contourf(xx,yy,Z_svm,alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k')
plt.title("Linear SVM (Straight boundary)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()