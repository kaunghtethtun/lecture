from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Generate slightly overlapping data
X, y = make_blobs(n_samples=200, centers=2,
                  cluster_std=2.0, random_state=42)

# Train original SVM
model = SVC(kernel='linear', C=1)
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_[0]

sv_indices = model.support_

# -----------------------------
# Remove ONE SUPPORT VECTOR
# -----------------------------
remove_sv_index = sv_indices[0]

X_sv_removed = np.delete(X, remove_sv_index, axis=0)
y_sv_removed = np.delete(y, remove_sv_index)

model_sv_removed = SVC(kernel='linear', C=1)
model_sv_removed.fit(X_sv_removed, y_sv_removed)

w_sv = model_sv_removed.coef_[0]
b_sv = model_sv_removed.intercept_[0]

# -----------------------------
# Remove ONE FAR-AWAY POINT
# -----------------------------
all_indices = np.arange(len(X))
non_sv_indices = np.setdiff1d(all_indices, sv_indices)
remove_far_index = non_sv_indices[0]

X_far_removed = np.delete(X, remove_far_index, axis=0)
y_far_removed = np.delete(y, remove_far_index)

model_far_removed = SVC(kernel='linear', C=1)
model_far_removed.fit(X_far_removed, y_far_removed)

w_far = model_far_removed.coef_[0]
b_far = model_far_removed.intercept_[0]

# -----------------------------
# Function to draw margin lines
# -----------------------------
def plot_svm(ax, X, y, w, b, support_vectors=None, title=""):
    x_points = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    y_decision = -(w[0]*x_points + b) / w[1]
    margin = 1 / np.linalg.norm(w)

    y_margin_pos = y_decision + margin
    y_margin_neg = y_decision - margin

    ax.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
    ax.plot(x_points, y_decision)
    ax.plot(x_points, y_margin_pos, linestyle='--')
    ax.plot(x_points, y_margin_neg, linestyle='--')

    if support_vectors is not None:
        ax.scatter(support_vectors[:,0],
                   support_vectors[:,1],
                   s=150,
                   facecolors='none',
                   edgecolors='black')

    ax.set_title(title)

# -----------------------------
# Plot everything
# -----------------------------
fig, axes = plt.subplots(1,3, figsize=(18,5))

plot_svm(axes[0], X, y, w, b,
         model.support_vectors_,
         "Original SVM")

plot_svm(axes[1], X_sv_removed, y_sv_removed,
         w_sv, b_sv,
         model_sv_removed.support_vectors_,
         "After Removing Support Vector")

plot_svm(axes[2], X_far_removed, y_far_removed,
         w_far, b_far,
         model_far_removed.support_vectors_,
         "After Removing Far-Away Point")

plt.tight_layout()
plt.show()

print("Original # SV:", len(model.support_))
print("After Removing SV # SV:", len(model_sv_removed.support_))
print("After Removing Far Point # SV:", len(model_far_removed.support_))