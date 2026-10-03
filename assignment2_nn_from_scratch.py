"""
Assignment 2: Neural Network from Scratch on make_moons
=======================================================
Architecture:
    Input layer : 2 neurons (x1, x2)
    Hidden layer: 3 neurons, ReLU activation
    Output layer: 1 neuron, Sigmoid activation

Loss: Binary Cross-Entropy
Optimiser: Batch Gradient Descent, 1000 epochs
Dataset: sklearn.datasets.make_moons(n_samples=200, noise=0.2)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# 1. Generate Data
# -------------------------------------------------
np.random.seed(42)

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # shape (200, 1) for vectorised BCE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Transpose to shape (n_features, m) — standard from-scratch convention
Xtr = X_train.T  # (2, 160)
Ytr = y_train.T  # (1, 160)
Xte = X_test.T   # (2, 40)
Yte = y_test.T   # (1, 40)
m = Xtr.shape[1]

# -------------------------------------------------
# 2. Helper functions (activations)
# -------------------------------------------------
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# -------------------------------------------------
# 3. Parameter initialisation
# -------------------------------------------------
def initialize_parameters(n_x=2, n_h=3, n_y=1):
    """
    He initialisation for ReLU hidden layer, Xavier for sigmoid output.
    He keeps ReLU activations from dying / exploding at start.
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * np.sqrt(2. / n_x)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * np.sqrt(1. / n_h)
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# -------------------------------------------------
# 4. Forward Propagation
# -------------------------------------------------
def forward_propagation(X, params):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    Z1 = np.dot(W1, X) + b1      # (n_h, m)
    A1 = relu(Z1)                # (n_h, m)
    Z2 = np.dot(W2, A1) + b2     # (1, m)
    A2 = sigmoid(Z2)             # (1, m) predicted P(y=1)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

# -------------------------------------------------
# 5. Compute Loss (Binary Cross-Entropy)
# -------------------------------------------------
def compute_loss(A2, Y):
    eps = 1e-15  # avoid log(0)
    A2 = np.clip(A2, eps, 1 - eps)
    loss = -np.mean(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss

# -------------------------------------------------
# 6. Backward Propagation (manually derived gradients)
# -------------------------------------------------
def backward_propagation(params, cache, Y):
    """
    Derivation (for one batch, mean loss):
      L = -mean( y log(a2) + (1-y) log(1-a2) )
      dZ2 = A2 - Y  (sigmoid + BCE combine neatly), divided by m
      dW2 = dZ2 . A1^T / m  ... etc.
      dA1 = W2^T . dZ2
      dZ1 = dA1 * relu'(Z1)
      dW1 = dZ1 . X^T / m
    """
    mm = Y.shape[1]
    A1, A2, Z1, X = cache["A1"], cache["A2"], cache["Z1"], cache["X"]
    W2 = params["W2"]

    dZ2 = (A2 - Y) / mm            # (1, m)
    dW2 = np.dot(dZ2, A1.T)        # (1, n_h)
    db2 = np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)        # (n_h, m)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(dZ1, X.T)         # (n_h, 2)
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

# -------------------------------------------------
# 7. Gradient Descent update
# -------------------------------------------------
def update_parameters(params, grads, lr):
    params["W1"] -= lr * grads["dW1"]
    params["b1"] -= lr * grads["db1"]
    params["W2"] -= lr * grads["dW2"]
    params["b2"] -= lr * grads["db2"]
    return params

def predict(X, params):
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)

# -------------------------------------------------
# 8. Training loop
# -------------------------------------------------
# Choice of learning rate:
#   - make_moons with noise=0.2 is a small, shallow non-linear problem.
#   - lr = 0.1 is a standard mid-range choice for batch GD + BCE on this scale:
#       * lr = 1.0  -> often overshoots / oscillates (loss diverges or bounces)
#       * lr = 0.01 -> converges but needs many more than 1000 epochs (slow)
#       * lr = 0.1  -> smooth, steady decrease within 1000 epochs (see experiment table)
#   Xavier/He init keeps initial gradients O(1), so lr=0.1 is stable.
LEARNING_RATE = 0.1
EPOCHS = 1000

def train(X, Y, n_h=3, lr=LEARNING_RATE, epochs=EPOCHS, verbose=True):
    params = initialize_parameters(n_x=2, n_h=n_h, n_y=1)
    losses = []
    for i in range(epochs):
        A2, cache = forward_propagation(X, params)
        loss = compute_loss(A2, Y)
        losses.append(loss)
        grads = backward_propagation(params, cache, Y)
        params = update_parameters(params, grads, lr)
        if verbose and (i % 200 == 0 or i == epochs - 1):
            print(f"Epoch {i:4d} | Loss: {loss:.4f}")
    return params, losses

print("\n--- Training main model (hidden=3, lr=0.1, epochs=1000) ---")
params, losses = train(Xtr, Ytr, n_h=3, lr=LEARNING_RATE, epochs=EPOCHS)

train_pred = predict(Xtr, params)
test_pred = predict(Xte, params)
train_acc = accuracy_score(Ytr.ravel(), train_pred.ravel())
test_acc = accuracy_score(Yte.ravel(), test_pred.ravel())
print(f"\nFinal Train Accuracy: {train_acc*100:.2f}%")
print(f"Final Test Accuracy : {test_acc*100:.2f}%")
print(f"Final Train Loss    : {losses[-1]:.4f}")

# -------------------------------------------------
# 9a. Plot: Loss vs Epochs
# -------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(losses)
plt.title("Loss vs Epochs (lr=0.1, hidden=3)")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Assignment_2_NN_From_Scratch/loss_vs_epochs.png", dpi=150)
print("Saved loss_vs_epochs.png")

# -------------------------------------------------
# 9b. Plot: Decision Boundary
# -------------------------------------------------
def plot_decision_boundary(pred_fn, X_all, y_all, title, savepath):
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    grid = np.c_[xx.ravel(), yy.ravel()].T  # (2, N)
    Z = pred_fn(grid)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(7, 5))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all.ravel(),
                cmap=plt.cm.RdYlBu, edgecolors="k", s=30)
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    print(f"Saved {savepath}")

plot_decision_boundary(
    lambda g: predict(g, params),
    X, y,
    "Decision Boundary (2-3-1 NN, ReLU+Sigmoid, lr=0.1)",
    "Assignment_2_NN_From_Scratch/decision_boundary.png",
)

# -------------------------------------------------
# 10. Experiments: effect of LR and hidden size
# -------------------------------------------------
print("\n--- Experiment: varying learning rate (hidden=3) ---")
for lr in [0.01, 0.1, 1.0]:
    p, ls = train(Xtr, Ytr, n_h=3, lr=lr, epochs=EPOCHS, verbose=False)
    ta = accuracy_score(Ytr.ravel(), predict(Xtr, p).ravel())
    sa = accuracy_score(Yte.ravel(), predict(Xte, p).ravel())
    print(f"lr={lr:<5} | final loss={ls[-1]:.4f} | train_acc={ta*100:5.2f}% | test_acc={sa*100:5.2f}%")

print("\n--- Experiment: varying hidden neurons (lr=0.1) ---")
for nh in [2, 3, 5, 10]:
    p, ls = train(Xtr, Ytr, n_h=nh, lr=0.1, epochs=EPOCHS, verbose=False)
    ta = accuracy_score(Ytr.ravel(), predict(Xtr, p).ravel())
    sa = accuracy_score(Yte.ravel(), predict(Xte, p).ravel())
    print(f"hidden={nh:<3} | final loss={ls[-1]:.4f} | train_acc={ta*100:5.2f}% | test_acc={sa*100:5.2f}%")
