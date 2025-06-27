"""
Tiny wrapper around create_sliding_windows so we can feed PyTorch easily.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

from src.data_utils import sliding_windows

class WindowDataset(Dataset):
    def __init__(self,
                 full_inputs: np.ndarray,
                 full_targets: np.ndarray,
                 mu_x: np.ndarray | None = None,
                 std_x: np.ndarray | None = None,
                 mu_y: float | None = None,
                 std_y: float | None = None):
        """
        Parameters
        ----------
        full_inputs  : array shape (N, n_features)
        full_targets : array shape (N,)
        mu_x, std_x  : array shape (n_features,)  (calcolati sul train-set)
        mu_y, std_y  : scalari                        idem
        """
        # ---------------- sliding windows
        X_win, y_fin = sliding_windows(full_inputs, full_targets)

        # ---------------- optional normalisation
        if mu_x is not None and std_x is not None:
            X_win = (X_win - mu_x) / std_x
        if mu_y is not None and std_y is not None:
            y_fin = (y_fin - mu_y) / std_y

        # ---------------- drop windows with NaN / inf
        mask = np.isfinite(X_win).all(axis=(1, 2)) & np.isfinite(y_fin)
        X_win, y_fin = X_win[mask], y_fin[mask]

        # ---------------- tensors
        self.X = torch.from_numpy(X_win).float()
        self.y = torch.from_numpy(y_fin).float()

        # store stats for inverse transform (utile in evaluate)
        self.mu_x, self.std_x = mu_x, std_x
        self.mu_y, self.std_y = mu_y, std_y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]