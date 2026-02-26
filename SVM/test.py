from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=200, centers=2, random_state=42)
X, y = make_blobs(n_samples=200, centers=2, cluster_std=3, random_state=42)

plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Binary Classification Data")
plt.show()

from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X, y)

import numpy as np

w = model.coef_[0]
b = model.intercept_[0]

x_points = np.linspace(X[:,0].min(), X[:,0].max())
y_points = -(w[0]*x_points + b)/w[1]

plt.scatter(X[:,0], X[:,1], c=y)
plt.plot(x_points, y_points, 'r')
plt.title("SVM Decision Boundary")
plt.show()