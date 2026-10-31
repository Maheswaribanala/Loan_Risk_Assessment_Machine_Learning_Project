"""
smote_custom.py
-----------------
A from-scratch, dependency-light implementation of SMOTE
(Synthetic Minority Over-sampling Technique) that supports multi-class
targets.

NOTE ON WHY THIS EXISTS:
The spec calls for `imbalanced-learn`'s SMOTE. This project was built in
an offline sandbox with no internet access, so `pip install
imbalanced-learn` could not run. The algorithm itself is simple enough to
reimplement faithfully with numpy + scikit-learn's NearestNeighbors, so
that's what this module does — it is a drop-in behavioural equivalent.

>>> ON A MACHINE WITH INTERNET ACCESS <<<
Simply `pip install imbalanced-learn` and swap:
    from smote_custom import SMOTE
for:
    from imblearn.over_sampling import SMOTE
The rest of train_models.py works unchanged since the .fit_resample(X, y)
interface is matched.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()

        X_resampled = [X]
        y_resampled = [y]

        for cls, count in zip(classes, counts):
            n_to_generate = majority_count - count
            if n_to_generate <= 0:
                continue

            X_cls = X[y == cls]
            k = min(self.k_neighbors, len(X_cls) - 1)
            if k < 1:
                # Too few samples in this class to interpolate; duplicate instead
                idx = rng.randint(0, len(X_cls), n_to_generate)
                synthetic = X_cls[idx]
            else:
                nn = NearestNeighbors(n_neighbors=k + 1).fit(X_cls)
                _, neighbor_idx = nn.kneighbors(X_cls)

                synthetic = np.zeros((n_to_generate, X.shape[1]))
                for i in range(n_to_generate):
                    sample_idx = rng.randint(0, len(X_cls))
                    neighbor_choice = rng.randint(1, k + 1)  # skip self (index 0)
                    neighbor = X_cls[neighbor_idx[sample_idx, neighbor_choice]]
                    base = X_cls[sample_idx]
                    gap = rng.rand()
                    synthetic[i] = base + gap * (neighbor - base)

            X_resampled.append(synthetic)
            y_resampled.append(np.full(n_to_generate, cls))

        X_out = np.vstack(X_resampled)
        y_out = np.concatenate(y_resampled)

        # shuffle
        perm = rng.permutation(len(X_out))
        return X_out[perm], y_out[perm]
