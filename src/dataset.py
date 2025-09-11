import torch
from torch.utils.data import Dataset
import numpy as np

from src.data_utils import sliding_windows

class WindowDataset(Dataset):
    def __init__(self, full_inputs, full_targets, mu_x=None, std_x=None, mu_y=None, std_y=None):
        X_view, y_fin = sliding_windows(full_inputs, full_targets)
        mask = np.isfinite(X_view).all(axis=(1, 2)) & np.isfinite(y_fin)

        self.X_view = X_view
        self.y_fin  = y_fin
        self.idx    = np.nonzero(mask)[0].astype(np.int64)

        self.mu_x, self.std_x = mu_x, std_x
        self.mu_y, self.std_y = mu_y, std_y

    def __len__(self):
        return int(self.idx.shape[0])

    def __getitem__(self, i):
        k = int(self.idx[i])
        Xw = self.X_view[k]
        yw = self.y_fin[k]

        # normalizzazione on-the-fly
        if self.mu_x is not None and self.std_x is not None:
            Xw = (Xw - self.mu_x) / self.std_x
        if self.mu_y is not None and self.std_y is not None:
            yw = (yw - self.mu_y) / self.std_y

        return torch.from_numpy(np.asarray(Xw, dtype=np.float32)), torch.tensor(yw, dtype=torch.float32)