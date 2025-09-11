"""
Top-level script – run this file only.
NOW:
1. Data loading, augmentation & split by cycle          <--- MOD
2. Dataset / Dataloader creation
3. ODE reference solution (multi-cycle)                 <--- MOD
4. Model training (standard & physics-informed)
5. Evaluation, FLOPs, and final plot
"""

# Standard libraries
import os
import time
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

# Local modules
from src.config import *
from src.data_utils import load_csv, compute_powers, solve_reference_ode, seed_everything, augment_cycle
from src.dataset import WindowDataset
from src.models import LSTMModel
from src.train import train_model

# Set random seed for reproducibility
seed = seed_everything(SEED)

# Optional: measure FLOPs if thop is available
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

def predict(model, dl, device):
    preds, gts = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)

Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

def main():
    # -------------------------------------------------------------- data loading
    cols   = load_csv()
    P_orig = compute_powers(cols["Id"], cols["Iq"])

    t_base     = cols["t"]
    dt         = t_base[1] - t_base[0]
    cycle_span = t_base[-1] - t_base[0]

    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []

    cycles_t  .append(t_base.copy())
    cycles_P  .append(P_orig.copy())
    cycles_Tbp.append(cols["Tbp"].copy())
    cycles_Tjr.append(cols["Tjr"].copy())

    for i in range(AUG_CYCLES):
        P_aug, Tbp_aug, Tjr_aug = augment_cycle(P_orig, cols["Tbp"], cols["Tjr"])
        t_shift = (i + 1) * (cycle_span + dt)
        cycles_t  .append(t_base + t_shift)
        cycles_P  .append(P_aug)
        cycles_Tbp.append(Tbp_aug)
        cycles_Tjr.append(Tjr_aug)

    n_cycles = len(cycles_t)
    n_train  = int(TRAIN_FRAC * n_cycles)
    n_val    = int(VAL_FRAC   * n_cycles)
    n_test   = n_cycles - n_train - n_val

    idx_train = list(range(0,                n_train))
    idx_val   = list(range(n_train,          n_train + n_val))
    idx_test  = list(range(n_train + n_val,  n_cycles))

    PAD = WINDOW_SIZE - 1

    def concat_with_pad(arr_list):
        if len(arr_list) == 0:
            return np.empty(0, dtype=float)
        pad = np.full(PAD, np.nan, dtype=arr_list[0].dtype)
        return np.concatenate([np.concatenate([a, pad]) for a in arr_list])[:-PAD]

    
    t_train   = concat_with_pad([cycles_t[i]   for i in idx_train])
    t_val     = concat_with_pad([cycles_t[i]   for i in idx_val  ])
    t_test    = concat_with_pad([cycles_t[i]   for i in idx_test ])

    P_train   = concat_with_pad([cycles_P[i]   for i in idx_train])
    P_val     = concat_with_pad([cycles_P[i]   for i in idx_val  ])
    P_test    = concat_with_pad([cycles_P[i]   for i in idx_test ])

    Tbp_train = concat_with_pad([cycles_Tbp[i] for i in idx_train])
    Tbp_val   = concat_with_pad([cycles_Tbp[i] for i in idx_val  ])
    Tbp_test  = concat_with_pad([cycles_Tbp[i] for i in idx_test ])

    Tjr_train = concat_with_pad([cycles_Tjr[i] for i in idx_train])
    Tjr_val   = concat_with_pad([cycles_Tjr[i] for i in idx_val  ])
    Tjr_test  = concat_with_pad([cycles_Tjr[i] for i in idx_test ])

    # -------------------------------------------------------------- plot input signals
    plt.figure()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_title("Input and output signals")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Temperature [°C]")
    ax2.set_ylabel("Power [W]")
    ax1.plot(t_base, cols["Tbp"], color='C1', label="Tbp")
    ax1.plot(t_base, cols["Tjr"], color='C2', label="Tjr (ground-truth)")
    ax2.plot(t_base, P_orig, color='C0', label="Pnet", alpha=0.5)
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    ax1.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path(PLOT_PATH), "signals.png"), dpi=300)

    exit()

    # -------------------------------------------------------------- dataset/dataloader
    X_train_raw = np.column_stack([P_train, Tbp_train])

    # ---------- helper
    def make_concat_dataset(idxs, mu_x, std_x, mu_y, std_y):
        ds_list = []
        for i in idxs:
            X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
            y = cycles_Tjr[i]
            ds_list.append(
                WindowDataset(X, y, mu_x, std_x, mu_y, std_y)
            )
        return ConcatDataset(ds_list)

    mu_x = np.nanmean(X_train_raw, axis=0)
    std_x = np.nanstd(X_train_raw, axis=0)

    mu_y = np.nanmean(Tjr_train)
    std_y = np.nanstd(Tjr_train)

    ds_train = make_concat_dataset(idx_train, mu_x, std_x, mu_y, std_y)
    ds_val   = make_concat_dataset(idx_val, mu_x, std_x, mu_y, std_y)
    ds_test  = make_concat_dataset(idx_test, mu_x, std_x, mu_y, std_y)

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)


    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Integrating ODE on each test cycle...")
    T_ode_list, ode_time_tot = [], 0.0
    for i in idx_test:
        T_i, ode_time = solve_reference_ode(cycles_t[i], cycles_P[i], cycles_Tbp[i])
        T_ode_list.append(T_i)
        ode_time_tot += ode_time
    # ---> padding identico a quello usato per i segnali DL
    T_ode = concat_with_pad(T_ode_list)
    print(f"ODE integration completed in {ode_time_tot:.2f} s (sum over {n_test} cycles)\n")

    # -------------------------------------------------------------- model A: pure data
    print("Training standard LSTM model (no physics)...")
    model_std = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    t0 = time.perf_counter()
    model_std = train_model(model_std, dl_train, dl_val, lambda_phys=0.0, device=device)
    std_train_time = time.perf_counter() - t0
    if THOP_AVAILABLE:
        dummy = torch.randn(1, WINDOW_SIZE, INPUT_SIZE).to(device)
        flops_std, params_std = profile(model_std, inputs=(dummy,), verbose=False)
        print(f"LSTM std  : {flops_std:.2e} FLOPs, {params_std:.0f} params")
    else:
        print("Install `thop` to measure FLOPs for the standard model.")
    print("Standard LSTM model training completed.\n")

    # -------------------------------------------------------------- model B: physics-informed
    print("Training physics-informed LSTM model...")
    model_pi = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
    t0 = time.perf_counter()
    model_pi = train_model(model_pi, dl_train, dl_val, lambda_phys=LAMBDA_PHYS, device=device)
    pi_train_time = time.perf_counter() - t0
    if THOP_AVAILABLE:
        dummy = torch.randn(1, WINDOW_SIZE, INPUT_SIZE).to(device)
        flops_pi,  params_pi  = profile(model_pi, inputs=(dummy,), verbose=False)
        print(f"LSTM phys : {flops_pi :.2e} FLOPs, {params_pi :.0f} params")
    else:
        print("Install `thop` to measure FLOPs for the physics-informed model.")
    print("Physics-informed LSTM model training completed.\n")

    # -------------------------------------------------------------- evaluation
    print("Evaluating models on the test set...")

    # 1) Forward pass on the whole test set ---------------------------------------
    #    ───────────────────────────────────────────────────────────────────────────
    yhat_std, y_gt = predict(model_std, dl_test, device)    # y_gt already aligned
    yhat_pi,  _    = predict(model_pi,  dl_test, device)

    # Undo the normalisation applied in WindowDataset
    yhat_std = mu_y + std_y * yhat_std
    y_gt     = mu_y + std_y * y_gt
    yhat_pi  = mu_y + std_y * yhat_pi

    # 2) Mean-square error --------------------------------------------------------
    mse_std = np.mean((yhat_std - y_gt)**2)
    mse_pi  = np.mean((yhat_pi  - y_gt)**2)

    # 3) ODE reference on the *raw* test samples ----------------------------------
    #    Cut away the first WINDOW_SIZE-1 points *once*, then drop NaN samples.
    start            = WINDOW_SIZE - 1
    t_full           = t_test[start:]
    T_gt_full        = Tjr_test[start:]
    T_ode_full       = T_ode[start:]

    mask_full        = ~np.isnan(T_gt_full)                 # sample-wise mask
    t_valid          = t_full[mask_full]
    T_gt_valid       = T_gt_full[mask_full]
    T_ode_valid      = T_ode_full[mask_full]

    mse_ode          = np.mean((T_ode_valid - T_gt_valid)**2)

    # 4) Build time stamps for *each* prediction ----------------------------------
    #    We must mimic the exact masking performed inside WindowDataset:
    from src.data_utils import sliding_windows          # helper that builds windows :contentReference[oaicite:0]{index=0}

    t_pred_list = []
    for i in idx_test:                                  # loop over every test cycle
        t_cycle   = cycles_t[i]
        P_cycle   = cycles_P[i]
        Tbp_cycle = cycles_Tbp[i]
        Tjr_cycle = cycles_Tjr[i]

        # Time stamp of every window *end* in the cycle
        t_end = t_cycle[WINDOW_SIZE - 1:]

        # Recreate windows and mask “good” ones (no NaN) – same rule as WindowDataset
        X_cycle = np.column_stack([P_cycle, Tbp_cycle])
        X_win, y_fin = sliding_windows(X_cycle, Tjr_cycle)
        mask = np.isfinite(X_win).all(axis=(1, 2)) & np.isfinite(y_fin)   # :contentReference[oaicite:1]{index=1}

        t_pred_list.append(t_end[mask])

    t_pred = np.concatenate(t_pred_list)                # len(t_pred) == len(yhat_std)

    print("Evaluation completed.\n")

    # -------------------------------------------------------------- plots

    # 1) Temperature comparison ----------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.title("Test set – Tjr comparison (all cycles)")
    plt.plot(t_valid, T_gt_valid,            label="Ground-truth Tjr", linewidth=1)
    plt.plot(t_valid, T_ode_valid, "--",     label="ODE",              linewidth=1)
    plt.plot(t_pred,  yhat_std,     ":",     label="LSTM",             linewidth=1)
    plt.plot(t_pred,  yhat_pi,      ":",     label="PI-LSTM",          linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path(PLOT_PATH), "test_results.png"), dpi=PLOT_DPI)

    # 2) Error curves --------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.title("Error plots – ODE vs LSTM models")
    plt.plot(t_valid, T_ode_valid - T_gt_valid, label="ODE error",     linewidth=1)
    plt.plot(t_pred,  yhat_std - y_gt,          label="LSTM error",    linewidth=1)
    plt.plot(t_pred,  yhat_pi  - y_gt,          label="PI-LSTM error", linewidth=1)
    plt.xlabel("Time [s]")
    plt.ylabel("ΔT [°C]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(Path(PLOT_PATH), "error_comparison.png"), dpi=PLOT_DPI)

if __name__ == "__main__":
    main()
