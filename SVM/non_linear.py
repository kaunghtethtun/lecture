from sklearn.datasets import make_circles
import matplotlib.pyplot as plt

X, y = make_circles(n_samples=300, factor=0.5, noise=0.1)

plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.title("Nonlinear Dataset (Circles)")
plt.show()

from sklearn.svm import SVC

# model_linear = SVC(kernel='linear')
# model_linear.fit(X, y)

# plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
# plt.title("Linear SVM (Fails)")
# plt.show()

model_rbf = SVC(kernel='rbf', gamma=1)
model_rbf.fit(X, y)

import numpy as np

# Create mesh grid
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
plt.title("RBF Kernel SVM (Nonlinear Boundary)")
plt.show()

for gamma in [0.1, 1, 10]:
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(X, y)
    print("Gamma =", gamma)