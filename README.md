# Assignment 2 — Neural Network from Scratch (make_moons)

## How to run
```bash
python Assignment_2_NN_From_Scratch/assignment2_nn_from_scratch.py
# or open Assignment_2_NN_From_Scratch/Assignment_2_Analysis.ipynb in Jupyter/VS Code
```
Requires: `numpy`, `matplotlib`, `scikit-learn`.

## What was built (all 7 steps)
1. **Data:** `make_moons(n_samples=200, noise=0.2)`, 80/20 train/test split (`random_state=42`) → 160 train / 40 test.
2. **Network (NumPy only):** 2 → 3 (ReLU) → 1 (Sigmoid). He init for hidden, Xavier for output.
3. **Forward prop:** `Z1=W1·X+b1, A1=ReLU(Z1), Z2=W2·A1+b2, A2=σ(Z2)`.
4. **Loss:** Binary Cross-Entropy with epsilon clipping.
5. **Backprop (manual):** `dZ2=(A2−Y)/m → dW2,db2 → dA1=W2ᵀ·dZ2 → dZ1=dA1⊙ReLU′(Z1) → dW1,db1`.
6. **Gradient descent:** `lr=0.1`, 1000 epochs batch GD.
   - *Why 0.1:* 0.01 converges too slowly (loss ~0.43 after 1000 epochs); 1.0 is fastest on this seed but risks overshoot generally; 0.1 gives smooth stable convergence — the standard speed/stability trade-off for a shallow BCE net.
7. **Visuals:** `loss_vs_epochs.png`, `decision_boundary.png` (in this folder).

## Results
- **Main model (hidden=3, lr=0.1, 1000 epochs): Train 84.38%, Test 82.50%, final loss ~0.29.**
- Varying LR (hidden=3): `0.01 → 80.00% test`, `0.1 → 82.50%`, `1.0 → 92.50%`.
- Varying hidden size (lr=0.1): `2 → 87.50%`, `3 → 82.50%`, `5 → 87.50%`, `10 → 82.50%`.

## Interpretation (submission Q3)
- **Learning rate:** too small underfits in the epoch budget; 0.1 converges steadily; 1.0 reaches the lowest loss here but is aggressive — use with caution (decay/clipping) as it can diverge on other inits.
- **Hidden neurons:** 2–5 units already capture the moons' non-linearity. 3 is the minimal spec and slightly under-capacity on noisy overlap; 5 fits best; 10 gives no further gain on 160 training points (capacity plateau, harder optimisation with plain GD).

## Files
| File | Purpose |
|---|---|
| `assignment2_nn_from_scratch.py` | Complete runnable implementation + experiments |
| `Assignment_2_Analysis.ipynb` | Analysis notebook (submission) |
| `loss_vs_epochs.png` | Loss vs epochs plot |
| `decision_boundary.png` | Decision boundary plot |
