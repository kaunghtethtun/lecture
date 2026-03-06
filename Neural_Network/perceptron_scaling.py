import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. DATA SCALING HELPERS ---
def min_max_scaling(data):
    return (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0))

def z_score_standardization(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# --- 2. PERCEPTRON CLASS ---
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10, verbose=False):
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose

    def activation(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        for epoch in range(self.epochs):
            if self.verbose:
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
            for i in range(len(X)):
                # z = w^T * x + b
                z = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(z)

                # Update rule: w = w + lr * (error) * x
                error = y[i] - y_pred
                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error

                if self.verbose:
                    print(
                        f"  Sample {i + 1}: x={X[i]}, y={y[i]}, z={z:.4f}, "
                        f"pred={y_pred}, error={error}, "
                        f"w={np.round(self.weights, 4)}, b={self.bias:.4f}"
                    )


def train_and_report(name, X, y, lr=0.1, epochs=10, verbose=False):
    print(f"\n=== {name} ===")
    model = Perceptron(input_size=X.shape[1], lr=lr, epochs=epochs, verbose=verbose)
    model.fit(X, y)
    print(f"Final Weights ({name}): {np.round(model.weights, 6)}")
    print(f"Final Bias ({name}): {round(model.bias, 6)}")
    return model


def plot_data(ax, X, y, title):
    class_0 = y == 0
    class_1 = y == 1
    ax.scatter(X[class_0, 0], X[class_0, 1], c="tomato", label="Class 0", s=70)
    ax.scatter(X[class_1, 0], X[class_1, 1], c="royalblue", label="Class 1", s=70)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(alpha=0.3)


def plot_decision_boundary(ax, model, X, y, title):
    plot_data(ax, X, y, title)

    w0, w1 = model.weights
    if abs(w1) < 1e-12:
        x_vertical = -model.bias / w0 if abs(w0) > 1e-12 else 0.0
        ax.axvline(x_vertical, color="black", linestyle="--", label="Decision boundary")
    else:
        x_vals = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 200)
        y_vals = -(w0 * x_vals + model.bias) / w1
        ax.plot(x_vals, y_vals, "k--", label="Decision boundary")

    ax.legend()

# --- 3. EXECUTION EXAMPLE ---
# Sample Data: [Feature 1, Feature 2]
raw_data = np.array([[10, 5000], [2, 1000], [8, 4500], [1, 800]])
labels = np.array([1, 0, 1, 0]) # Binary classification

# Scale data
X_scaled = min_max_scaling(raw_data)
X_zscore = z_score_standardization(raw_data)

# Train and print results
# verbose=True prints step-by-step updates for each sample and epoch.
model_raw = train_and_report("Raw Data", raw_data, labels, verbose=True)
model_minmax = train_and_report("Min-Max Scaled", X_scaled, labels, verbose=True)
model_zscore = train_and_report("Z-Score Standardized", X_zscore, labels, verbose=True)

# Visualize datasets and decision boundaries
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Perceptron: Data Scaling and Decision Boundaries", fontsize=14)

plot_data(axes[0, 0], raw_data, labels, "Raw Data")
plot_data(axes[0, 1], X_scaled, labels, "Min-Max Scaled Data")
plot_data(axes[0, 2], X_zscore, labels, "Z-Score Standardized Data")

plot_decision_boundary(axes[1, 0], model_raw, raw_data, labels, "Perceptron on Raw Data")
plot_decision_boundary(axes[1, 1], model_minmax, X_scaled, labels, "Perceptron on Min-Max Data")
plot_decision_boundary(axes[1, 2], model_zscore, X_zscore, labels, "Perceptron on Z-Score Data")

plt.tight_layout()
if "agg" in plt.get_backend().lower():
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "perceptron_scaling_visual.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved visualization to: {output_path}")
else:
    plt.show()
