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
    CSV_DIR, RTH_C, RTH_V, C_TH, T_ENV, R_DSON,
    WINDOW_SIZE, SCALE_STD, NOISE_STD, TEMP_OFFSET_STD, JITTER_SAMPLES
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

from glob import glob
from src.config import CSV_DIR, CSV_GLOB
# …

def list_csv_files() -> list[str]:
    """Restituisce la lista completa dei CSV in `CSV_DIR`."""
    return sorted(glob(os.path.join(CSV_DIR, CSV_GLOB)))

# ---------------------------------------------------------------------- loaders
def load_single_csv(path) -> dict[str, np.ndarray]:
    """Return a dict with all useful columns, already time-sorted."""
    data = np.loadtxt(Path(path), delimiter=",", skiprows=1)

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

def load_all_csvs() -> list[dict[str, np.ndarray]]:
    """Carica tutti i CSV e restituisce una lista di dict → colonne."""
    datasets = []
    for f in list_csv_files():
        cols = load_single_csv(f)       # ex-load_csv(path)
        datasets.append(cols)
    return datasets


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


from numpy.lib.stride_tricks import sliding_window_view

def sliding_windows(data_in: np.ndarray, data_out: np.ndarray):
    """
    Convert (T, N_features) + (T,) → (N_windows, WINDOW_SIZE, N_features) + (N_windows,).
    Vectorized implementation using NumPy sliding_window_view.
    """
    # data_in: (T, F), data_out: (T,)
    # sliding along time axis (axis=0):
    # windows shape → (T-WINDOW_SIZE+1, WINDOW_SIZE, F)
    # windows: dovrebbe essere (N_windows, WINDOW_SIZE, N_features)
    windows = sliding_window_view(data_in, window_shape=WINDOW_SIZE, axis=0)
    # Se invece è (N_windows, N_features, WINDOW_SIZE), inverti gli ultimi due assi
    if windows.ndim == 3 and windows.shape[1] != WINDOW_SIZE:
        windows = windows.transpose(0, 2, 1)
    # targets aligned to end of each window
    targets = data_out[WINDOW_SIZE - 1 :]
    return windows, targets

# ------------------------------------------------------------------- augmentations
def augment_cycle(
    P:    np.ndarray,
    Tbp:  np.ndarray,
    Tjr:  np.ndarray,
    *,
    scale_std:   float = SCALE_STD,
    noise_std:   float = NOISE_STD,
    offset_std:  float = TEMP_OFFSET_STD,
    jitter_max:  int   = JITTER_SAMPLES,
    rng=np.random.default_rng(),
):
    """
    Physical-aware duty-cycle augmentation.

    1) **Jitter temporale** – roll di ±`jitter_max` campioni applicato
       in blocco a P, Tbp e Tjr (mantiene la coerenza fra canali).
    2) **Offset termico quasi-statico** – sposta Tbp e Tjr di ΔT ~ N(0, σ²)
       per simulare variazioni lente dell’ambiente.
    3) **Scaling di potenza** – fattore N(1, scale_std²).
    4) **Rumore a somma-zero** su P – preserva l’integrale ∫P dt (energia),
       quindi non altera la dinamica globale del transiente.
    5) **Rumore bianco** su Tbp/Tjr come prima.

    Restituisce: tuple (P_aug, Tbp_aug, Tjr_aug) con stessa shape originale.
    """
    # ---------- 1) jitter temporale (roll)
    shift = int(rng.integers(-jitter_max, jitter_max + 1))
    P_aug, Tbp_aug, Tjr_aug = (
        np.roll(P,   shift),
        np.roll(Tbp, shift),
        np.roll(Tjr, shift),
    )

    # ---------- 2) offset termico quasi-statico
    dT = rng.normal(0.0, offset_std)
    Tbp_aug += dT
    Tjr_aug += dT

    # ---------- 3) scaling globale di potenza
    amp = rng.normal(1.0, scale_std)          # ≈ ±5 %
    P_aug *= amp

    # ---------- 4) rumore a somma-zero (conserva energia)
    noise_P = rng.normal(0.0, noise_std * P_aug.std(), size=P_aug.shape)
    noise_P -= noise_P.mean()                 # ∑ noise_P = 0
    P_aug += noise_P

    # ---------- 5) rumore su temperature (come prima)
    noise_bp = rng.normal(0.0, noise_std * Tbp_aug.std(), size=Tbp_aug.shape)
    noise_jr = rng.normal(0.0, noise_std * Tjr_aug.std(), size=Tjr_aug.shape)
    Tbp_aug += noise_bp
    Tjr_aug += noise_jr

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
