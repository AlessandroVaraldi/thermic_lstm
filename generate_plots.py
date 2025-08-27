#!/usr/bin/env python
# scripts/generate_plots.py
"""
Post-processing utility: loads the best data-driven (STD) and physics-informed
(PI) checkpoints and generates a full suite of plots:

    • time-series with confidence intervals
    • reliability (calibration) curves
    • |ε| vs σ scatter
    • normalised-error histogram
    • attention heat-map
    • cross-model comparison plots (timeseries, calibration, hist, bars)

All PNGs are saved to ``config.PLOT_PATH`` with self-explanatory names.

Usage
-----
python -m scripts.generate_plots \
    --ckpt_std checkpoints/std_best.pth \
    --ckpt_pi  checkpoints/pi_best.pth \
    --device   cuda:0
"""

from __future__ import annotations

# --------------------------------------------------------------------------- std-lib
import argparse
from pathlib import Path
import pickle
import random

# --------------------------------------------------------------------------- third-party
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import torch
import torch.serialization as _ts
from torch.utils.data import DataLoader

# --------------------------------------------------------------------------- project
from src.config import (
    TRAIN_RATIO, VAL_RATIO, WINDOW_SIZE, BATCH_SIZE,
    PLOT_PATH, PLOT_DPI, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
)
from src.data_utils import load_all_csvs, compute_powers, predict_mc
from src.dataset    import WindowDataset
from src.models     import LSTMModel

# --------------------------------------------------------------------------- colour palette
COLOR_BASELINE  = "#4596d0"   # cool blue-teal – reference / STD
COLOR_HIGHLIGHT = "#c42520"   # hot red       – PI

plt.rcParams["axes.prop_cycle"] = cycler(color=[COLOR_BASELINE, COLOR_HIGHLIGHT])
plt.rcParams.update({
    "font.size"         : 14,   # base size
    "axes.titlesize"    : 16,
    "axes.labelsize"    : 14,
    "legend.fontsize"   : 13,
    "xtick.labelsize"   : 12,
    "ytick.labelsize"   : 12,
})


# --------------------------------------------------------------------------- utils
def common_edges(arrays: list[np.ndarray], n_bins: int = 40, rule: str = "fd") -> np.ndarray:
    """
    Compute common histogram edges for *all* arrays.
    If `rule == "fd"`, use Freedman-Diaconis to derive `n_bins` automatically.
    """
    data = np.concatenate(arrays)
    if isinstance(rule, str) and rule.lower() == "fd":
        q25, q75 = np.percentile(data, [25, 75])
        iqr = q75 - q25
        bin_w = 2 * iqr * data.size ** (-1 / 3) or 1.0
        n_bins = max(10, int(np.ceil((data.max() - data.min()) / bin_w)))
    return np.linspace(data.min(), data.max(), n_bins + 1)

def _pred_colour(tag: str) -> str:
    return COLOR_BASELINE if tag.upper().startswith("STD") else COLOR_HIGHLIGHT

# --------------------------------------------------------------------------- dataset rebuild
def make_datasets():
    """
    Reconstruct train/val/test splits from raw CSVs to recover the *exact*
    normalisation statistics used during training.
    """
    records = load_all_csvs()
    t_all, feats, target = [], [], []

    for rec in records:
        P   = compute_powers(rec["Id"], rec["Iq"])           # (T,)
        Tbp = rec["Tbp"]
        inp = np.stack([P, Tbp], axis=1)                     # (T, 2)

        t_all.append(rec["t"])
        feats.append(inp)
        target.append(rec["Tjr"])

    t_all  = np.concatenate(t_all)
    feats  = np.concatenate(feats)
    target = np.concatenate(target)

    N      = len(t_all)
    idx_tr = int(N * TRAIN_RATIO)
    idx_va = int(N * (TRAIN_RATIO + VAL_RATIO))

    mu_x, std_x = feats[:idx_tr].mean(0), feats[:idx_tr].std(0)
    mu_y, std_y = target[:idx_tr].mean(),  target[:idx_tr].std()

    def _mk(x, y):
        return WindowDataset(x, y, mu_x, std_x, mu_y, std_y)

    train_ds = _mk(feats[:idx_tr],      target[:idx_tr])
    val_ds   = _mk(feats[idx_tr:idx_va], target[idx_tr:idx_va])
    test_ds  = _mk(feats[idx_va:],      target[idx_va:])

    return train_ds, val_ds, test_ds, (mu_x, std_x, mu_y, std_y)

# --------------------------------------------------------------------------- plotting primitives
def plot_timeseries(t, y_true, mu, sigma, tag):
    c_pred = _pred_colour(tag)
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_true, lw=1.0, c="k", label="Truth")
    plt.plot(t, mu,     lw=1.0, c=c_pred, label=f"{tag} μ")
    for k, alpha, lab in [(1.0, .30, "±1σ"), (1.96, .15, "±1.96σ")]:
        plt.fill_between(t, mu - k * sigma, mu + k * sigma,
                         alpha=alpha, color=c_pred, label=lab)
    plt.xlabel("Sample index"); plt.ylabel("T [°C]")
    plt.title(f"Prediction – {tag}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/{tag}_timeseries_ci.png", dpi=PLOT_DPI)
    plt.close()

def plot_calibration(std_pred, errs, tag, n_bins=15, level=0.68):
    """Single-model reliability diagram."""
    edges = common_edges([std_pred], n_bins)
    emp, nom = [], []
    k = 1.0 if np.isclose(level, .68) else 1.96
    for i in range(n_bins):
        m = (std_pred >= edges[i]) & (std_pred < edges[i+1])
        if m.any():
            emp.append((errs[m] <= k * std_pred[m]).mean())
            nom.append(level)
    plt.figure()
    plt.plot(nom, emp, "o-")
    plt.plot([0, 1], [0, 1], "--k", lw=.8)
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical")
    plt.title(f"Calibration – {tag} ({int(level*100)} %)")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/{tag}_calibration.png", dpi=PLOT_DPI)
    plt.close()

def plot_err_vs_sigma(std_pred, errs, tag, max_points=20_000):
    if len(errs) > max_points:
        idx = np.random.choice(len(errs), max_points, replace=False)
        std_pred, errs = std_pred[idx], errs[idx]
    plt.figure()
    plt.scatter(std_pred, errs, s=6, alpha=.25, c=_pred_colour(tag))
    plt.xlabel("Predicted σ [°C]"); plt.ylabel("|error| [°C]")
    plt.title(f"|ϵ| vs σ – {tag}"); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/{tag}_err_vs_sigma.png", dpi=PLOT_DPI)
    plt.close()

def plot_hist_normalised(errs, std_pred, tag, n_bins=60):
    z = errs / std_pred
    plt.figure()
    plt.hist(z, bins=n_bins, density=True, alpha=.7, color=_pred_colour(tag))
    plt.axvline( 1.96, ls="--", c="k"); plt.axvline(-1.96, ls="--", c="k")
    plt.xlabel("Normalised error  z = ϵ/σ"); plt.ylabel("pdf")
    plt.title(f"z distribution – {tag}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/{tag}_hist_z.png", dpi=PLOT_DPI)
    plt.close()

def plot_attention_heatmap(weights_all, tag):
    plt.figure(figsize=(8, 4))
    plt.imshow(weights_all, aspect="auto", interpolation="nearest", cmap="viridis")
    plt.colorbar(label="α (attention)")
    plt.xlabel("Lag in window"); plt.ylabel("Sample index")
    plt.title(f"Attention heat-map – {tag}")
    plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/{tag}_attention.png", dpi=PLOT_DPI)
    plt.close()

# --------------------------------------------------------------------------- comparison plots
def plot_compare_timeseries(res_std, res_pi, step=1):
    idx = slice(None, None, step)
    t   = np.arange(len(res_std["y_true"]))[idx]

    plt.figure(figsize=(8, 8))
    plt.plot(t, res_std["y_true"][idx], lw=.8, c="k", label="Truth")
    plt.plot(t, res_std["mu"][idx],     lw=.8, c=COLOR_BASELINE,  label="STD μ")
    plt.plot(t, res_pi ["mu"][idx],     lw=.8, c=COLOR_HIGHLIGHT, label="PI  μ")
    plt.fill_between(t,
                     res_pi["mu"][idx] - 1.96*res_pi["sigma"][idx],
                     res_pi["mu"][idx] + 1.96*res_pi["sigma"][idx],
                     color=COLOR_HIGHLIGHT, alpha=.12, label="PI 95 % CI")
    plt.xlabel("Time (s)"); plt.ylabel("T [°C]")
    plt.title("Prediction overlay – full test-set")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/COMPARE_timeseries_overlay.png", dpi=PLOT_DPI)
    plt.close()

def plot_compare_calibration(res_std, res_pi, n_bins=15, level=0.95):
    k = 1.96 if np.isclose(level, .95) else 1.0
    edges = common_edges([res_std["sigma"], res_pi["sigma"]], n_bins)

    def _emp(res):
        out = []
        for i in range(n_bins):
            m = (res["sigma"] >= edges[i]) & (res["sigma"] < edges[i+1])
            out.append((res["errs"][m] <= k * res["sigma"][m]).mean() if m.any() else np.nan)
        return np.array(out)

    emp_s, emp_p = _emp(res_std), _emp(res_pi)
    nom = np.full(n_bins, level)

    plt.figure()
    plt.plot(nom, emp_s, "o-",  label="STD", color=COLOR_BASELINE)
    plt.plot(nom, emp_p, "s--", label="PI",  color=COLOR_HIGHLIGHT)
    plt.plot([0, 1], [0, 1], "--k", lw=.8)
    plt.xlabel("Nominal coverage"); plt.ylabel("Empirical")
    plt.title(f"Reliability diagram – {int(level*100)} % interval")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/COMPARE_calibration.png", dpi=PLOT_DPI)
    plt.close()

def plot_compare_err_hist(res_std, res_pi, n_bins=40):
    edges = common_edges([res_std["errs"], res_pi["errs"]], n_bins)
    plt.figure(figsize=(8, 8))
    plt.hist(res_std["errs"], bins=edges, alpha=.5,
             label="STD", color=COLOR_BASELINE)
    plt.hist(res_pi ["errs"], bins=edges, alpha=.5,
             label="PI",  color=COLOR_HIGHLIGHT)
    plt.xlabel("|error| (°C)"); plt.ylabel("# samples")
    plt.title("Absolute-error distribution")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/COMPARE_err_hist.png", dpi=PLOT_DPI)
    plt.close()

def plot_compare_metrics_bar(res_std, res_pi):
    mse_s, mae_s = np.mean(res_std["errs"]**2), np.mean(res_std["errs"])
    mse_p, mae_p = np.mean(res_pi ["errs"]**2), np.mean(res_pi ["errs"])
    x = np.arange(2)
    plt.figure()
    plt.bar(x - 0.15, [mse_s, mae_s], width=0.3, label="STD", color=COLOR_BASELINE)
    plt.bar(x + 0.15, [mse_p, mae_p], width=0.3, label="PI",  color=COLOR_HIGHLIGHT)
    plt.xticks(x, ["MSE", "MAE"]); plt.ylabel("[°C²] / [°C]")
    plt.title("Point-error metrics")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{PLOT_PATH}/COMPARE_metrics_bar.png", dpi=PLOT_DPI)
    plt.close()

# --------------------------------------------------------------------------- attention helper
@torch.no_grad()
def get_attention_weights(model: LSTMModel, xb: torch.Tensor) -> torch.Tensor:
    outputs, (h_T, _) = model.lstm(xb)
    query   = h_T[-1]                                   # (B, H)
    scores  = torch.bmm(outputs, query.unsqueeze(2)).squeeze(2)
    weights = torch.nn.functional.softmax(scores, dim=1)
    return weights                                      # (B, W)

# --------------------------------------------------------------------------- checkpoint loader
def load_lstm_from_ckpt(ckpt_path: str, device: str) -> LSTMModel:
    """
    Return an `LSTMModel` no matter whether the file is a plain state-dict or
    a fully pickled model object.  Safe with PyTorch ≥ 2.6.
    """
    try:
        state_dict = torch.load(ckpt_path, map_location=device)         # weights_only=True
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
        model.load_state_dict(state_dict)
        return model
    except Exception:                                                   # fallback
        _ts.add_safe_globals({"src.models.LSTMModel": LSTMModel})
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(loaded, LSTMModel):
            return loaded.to(device)
        model = LSTMModel(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(device)
        state = loaded["state_dict"] if isinstance(loaded, dict) and "state_dict" in loaded else loaded
        model.load_state_dict(state)
        return model

# --------------------------------------------------------------------------- evaluation
def evaluate_model(tag, ckpt_path, test_dl, device, stats):
    """MC-dropout inference + individual plots."""
    model = load_lstm_from_ckpt(ckpt_path, device)
    model.eval()

    mu_hat, sigma_hat = predict_mc(model, test_dl, n_samples=50, device=device)

    _, _, mu_y, std_y = stats
    mu_hat  = mu_y + std_y * mu_hat
    sigma_hat = std_y * sigma_hat

    y_true = torch.cat([y for _, y in test_dl]).numpy()
    y_true = mu_y + std_y * y_true
    errs   = np.abs(mu_hat - y_true)

    t_idx = np.arange(len(y_true))
    plot_timeseries(t_idx, y_true, mu_hat, sigma_hat, tag)
    plot_calibration(sigma_hat, errs, tag)
    plot_err_vs_sigma(sigma_hat, errs, tag)
    plot_hist_normalised(errs, sigma_hat, tag)

    # attention heat-map
    weights_all = []
    for xb, _ in test_dl:
        w = get_attention_weights(model, xb.to(device))
        weights_all.append(w.cpu())
    plot_attention_heatmap(torch.cat(weights_all).numpy(), tag)

    print(f"[{tag}] plots saved to «{PLOT_PATH}»")

    return {"tag": tag,
            "y_true": y_true,
            "mu": mu_hat,
            "sigma": sigma_hat,
            "errs": errs}

# --------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_std", required=True, help="Path to best STD checkpoint")
    parser.add_argument("--ckpt_pi",  required=True, help="Path to best PI  checkpoint")
    parser.add_argument("--device",   default="cpu")
    parser.add_argument("--ts_step",  type=int, default=1,
                        help="Sub-sampling step for full time-series overlay")
    parser.add_argument("--bins",     type=int, default=40,
                        help="Histogram / calibration bin count")
    args = parser.parse_args()

    Path(PLOT_PATH).mkdir(exist_ok=True)

    _, _, test_ds, stats = make_datasets()
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    res_std = evaluate_model("STD", args.ckpt_std, test_dl, args.device, stats)
    res_pi  = evaluate_model("PI",  args.ckpt_pi,  test_dl, args.device, stats)

    # ---------- comparison plots ----------
    plot_compare_timeseries(res_std, res_pi, step=args.ts_step)
    plot_compare_calibration(res_std, res_pi, n_bins=args.bins)
    plot_compare_err_hist(res_std, res_pi, n_bins=args.bins)
    plot_compare_metrics_bar(res_std, res_pi)

if __name__ == "__main__":
    main()
