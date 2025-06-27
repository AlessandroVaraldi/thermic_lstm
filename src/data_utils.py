"""
I/O and pre-processing utilities.
All NumPy only – no PyTorch dependencies here.
"""

from __future__ import annotations

# Standard library imports
import os
import random
from pathlib import Path
from typing import Final

# Third-party imports
import numpy as np
import torch
from scipy.integrate import solve_ivp

# Local application imports
from src.config import (
    CSV_FILE,
    RTH_C,
    RTH_V,
    C_TH,
    R_DSON,
    TRAIN_RATIO,
    VAL_RATIO,
    WINDOW_SIZE,
    T_ENV,
    SCALE_STD,
    NOISE_STD
)
from src.phys_models import thermal_ode

def seed_everything(seed: int | None = None) -> int:
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little")

    # 1) Python & hash
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) NumPy
    np.random.seed(seed)

    # 3) PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4) cuDNN
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    return seed

# ---------------------------------------------------------------------- loaders
def load_csv() -> dict[str, np.ndarray]:
    """Return a dict with all useful columns, already time-sorted."""
    data = np.loadtxt(Path(CSV_FILE), delimiter=",", skiprows=1)

    columns = {
        "t":   data[:, 0],
        "Id":  data[:, 5],
        "Iq":  data[:, 6],
        "Tbp": data[:, 9],
        "Tjr": data[:, 12],        # ground-truth junction temperature
    }

    # sort by time just in case
    idx = np.argsort(columns["t"])
    for k in columns:
        columns[k] = columns[k][idx]

    return columns


# -------------------------------------------------------------- derived signals
def compute_powers(Id: np.ndarray, Iq: np.ndarray) -> np.ndarray:
    """Compute net power dissipated in the chip."""
    Ias       = np.sqrt(Id**2 + Iq**2)
    Ias_chip  = Ias / np.sqrt(2) / 3.0
    P_cond    = (Ias_chip**2) * R_DSON     # conduction loss
    P_net     = P_cond - 3.0               # minus constant gate losses
    return P_net


# --------------------------------------------------------------- ODE reference
def solve_reference_ode(t, P, Tbp) -> tuple[np.ndarray, float]:
    """
    Integrate the RC model once for the whole record.
    Returns simulated temperature and wall-clock time [s].
    """
    T0 = [Tbp[0]]
    ode = lambda t_, T_: thermal_ode(
        T_, t_, t, P, Tbp, np.full_like(t, T_ENV), RTH_C, RTH_V, C_TH
    )
    from time import perf_counter
    tic = perf_counter()
    sol = solve_ivp(ode, (t[0], t[-1]), T0, t_eval=t, method="RK45")
    runtime = perf_counter() - tic
    return sol.y[0], runtime


# ------------------------------------------------------------------- windows
def sliding_windows(data_in: np.ndarray, data_out: np.ndarray):
    """
    Convert (T,N_features) + (T,) → (N, window, features) + (N,).
    """
    X, y = [], []
    for i in range(len(data_in) - WINDOW_SIZE + 1):
        X.append(data_in[i : i + WINDOW_SIZE])
        y.append(data_out[i + WINDOW_SIZE - 1])
    return np.asarray(X), np.asarray(y)

# ------------------------------------------------------------------- augmentations
def augment_cycle(P, Tbp, Tjr,
                  scale_std=SCALE_STD,
                  noise_std=NOISE_STD,
                  rng=np.random.default_rng()):
    """Restituisce una copia del ciclo con semplici augmentations fisicamente plausibili."""
    amp      = rng.normal(1.0, scale_std)            # ±5 % amplitude scaling
    noise_bp = rng.normal(0.0, noise_std * Tbp.std(), size=Tbp.shape)
    noise_jr = rng.normal(0.0, noise_std * Tjr.std(), size=Tjr.shape)
    P_aug    = P   * amp
    Tbp_aug  = Tbp + noise_bp
    Tjr_aug  = Tjr + noise_jr
    return P_aug, Tbp_aug, Tjr_aug


@torch.no_grad()
def predict_mc(model, dataloader, n_samples=30, device="cpu"):
    """
    Monte-Carlo Dropout inference.

    Returns
    -------
    mean : np.ndarray, shape (N,)
    std  : np.ndarray, shape (N,)
    """
    model.eval()                     # BatchNorm resterebbe in eval (qui non lo usiamo)
    preds = []
    for _ in range(n_samples):
        single_pass = []
        for xb, _ in dataloader:
            xb = xb.to(device)
            yb = model(xb, mc_dropout=True)     # <-- abilita dropout
            single_pass.append(yb.cpu())
        preds.append(torch.cat(single_pass))
    stack = torch.stack(preds)        # (S, N)
    return stack.mean(0).numpy(), stack.std(0).numpy()

