#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qlstm_test.py — Evaluation & diagnostics for PI-LSTM INT8-QAT

• Re-uses the exact test split saved by qat.py (meta_qat.json),
  but intentionally DROPS any augmented cycles from that split (no test-time augmentation).
• Computes MSE/MAE/RMSE/R² in °C.
• Saves diagnostic plots into plots/:
    - test_parity.png
    - test_residual_hist.png
    - test_residual_vs_P_Tbp.png
    - test_overlay_cycle_{k}.png  (k = 0..N-1)
• Optionally writes a CSV with all test predictions.

Usage (examples)
  python qlstm_test.py --ckpt checkpoints/best_qat.pth --meta checkpoints/meta_qat.json
  python qlstm_test.py --device cuda --overlay-cycles 3 --save-csv plots/test_predictions.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# --- project-local imports
from src.config import PLOT_DPI, WINDOW_SIZE
from src.data_utils import load_all_csvs, compute_powers, sliding_windows, solve_reference_ode  # :contentReference[oaicite:3]{index=3}
from src.dataset import WindowDataset  # :contentReference[oaicite:4]{index=4}
from src.qat_int8 import LSTMModelInt8QAT  # :contentReference[oaicite:5]{index=5}

PLOTS_DIR = Path("qplots"); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _devtype(device: str | torch.device) -> str:
    s = str(device)
    return "cuda" if ("cuda" in s and torch.cuda.is_available()) else "cpu"

def _build_original_cycles():
    """
    Carica i CSV e costruisce SOLO i cicli originali (no augmentation).
    Restituisce liste parallele cycles_t, cycles_P, cycles_Tbp, cycles_Tjr.
    """
    datasets = load_all_csvs()  # ciascun elemento ha chiavi: t, Id, Iq, Tbp, Tjr  :contentReference[oaicite:6]{index=6}
    if len(datasets) == 0:
        raise FileNotFoundError("No CSV found. Check src.config.CSV_DIR/CSV_GLOB.")
    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []
    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])      # :contentReference[oaicite:7]{index=7}
        cycles_t.append(cols["t"])
        cycles_P.append(P)
        cycles_Tbp.append(cols["Tbp"])
        cycles_Tjr.append(cols["Tjr"])
    return cycles_t, cycles_P, cycles_Tbp, cycles_Tjr

class _Empty(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

def _build_concat_dataset_from_idxs(idxs, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y):
    if not idxs: return _Empty()
    ds_list = []
    for i in idxs:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])  # (T, 2)
        y = cycles_Tjr[i]                                  # (T,)
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))  # :contentReference[oaicite:8]{index=8}
    return ConcatDataset(ds_list)

@torch.no_grad()
def _eval_epoch(model, dl, device, mu_y, std_y, *, collect_last_step=False):
    """
    Valuta il modello e ritorna:
      mse (denormalizzato °C), y_pred (°C), y_true (°C), extras (facoltativi)
    Se collect_last_step=True, accumula P_last e Tbp_last per ciascuna finestra
    per plot diagnostici degli errori vs feature.
    """
    model.eval()
    tot, n = 0.0, 0
    preds, gts = [], []
    P_last, Tbp_last = [], []
    for xb, yb in dl:
        xb = xb.to(device); yb = yb.to(device)
        y_hat = model(xb)  # output normalizzato (B,)
        tot += F.mse_loss(y_hat, yb, reduction="sum").item()
        n   += yb.numel()
        # denorm in °C
        preds.append((y_hat.float().cpu().numpy() * std_y + mu_y))
        gts.append  ((yb.float().cpu().numpy()     * std_y + mu_y))
        if collect_last_step:
            x_last = xb[:, -1, :].float().cpu().numpy()  # (B,2) in unità normalizzate
            # de-normalizza rispetto a mu_x/std_x per avere P,Tbp reali quando servono nei plot
            # NB: li ricaveremo fuori dove conosciamo mu_x/std_x
            P_last.append(x_last[:, 0])
            Tbp_last.append(x_last[:, 1])
    yh = np.concatenate(preds) if preds else np.empty((0,), dtype=float)
    yt = np.concatenate(gts)   if gts   else np.empty((0,), dtype=float)
    extras = {}
    if collect_last_step:
        extras["x_last_norm_P"] = np.concatenate(P_last) if P_last else np.empty((0,), dtype=float)
        extras["x_last_norm_Tbp"] = np.concatenate(Tbp_last) if Tbp_last else np.empty((0,), dtype=float)
    return (tot/max(n,1)), yh, yt, extras

def main():
    ap = argparse.ArgumentParser("Test/diagnostics for PI-LSTM INT8-QAT")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best_qat.pth")
    ap.add_argument("--meta", type=str, default="checkpoints/meta_qat.json")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--overlay-cycles", type=int, default=3, help="numero di cicli di test da plottare come overlay")
    ap.add_argument("--save-csv", type=str, default="", help="opzionale: path CSV per salvare predizioni/GT di tutte le finestre di test")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    meta_path = Path(args.meta)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    assert meta_path.exists(), f"Meta JSON not found: {meta_path}"

    meta = json.loads(meta_path.read_text())
    mu_x  = np.array(meta["mu_x"], dtype=np.float32)
    std_x = np.array(meta["std_x"], dtype=np.float32) + 1e-6
    mu_y  = float(meta["mu_y"])
    std_y = float(meta["std_y"])
    split_saved = meta["splits"]  # dict con 'train','val','test' (liste di int)

    # 1) ricostruisci SOLO i cicli originali (no augmentation)
    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = _build_original_cycles()
    n_original = len(cycles_P)

    # 2) seleziona dal test salvato SOLO gli indici < n_original (droppa augmented)
    idx_test_raw = list(map(int, split_saved.get("test", [])))
    idx_test = [i for i in idx_test_raw if i < n_original]
    dropped = len(idx_test_raw) - len(idx_test)
    if dropped > 0:
        print(f"[info] Dropped {dropped} augmented cycles from saved test split (test-time has no augmentation).")

    # 3) dataset & dataloader di test
    ds_te = _build_concat_dataset_from_idxs(idx_test, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=0)

    # 4) ricostruisci il modello coerente con il training e carica pesi
    hidden  = int(meta.get("hidden", 16))
    layers  = int(meta.get("layers", 1))
    dropout = float(meta.get("dropout", 0.10))
    S_gate_q8  = int(meta.get("S_gate_q8", 32))
    S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))

    device = torch.device(args.device)
    model = LSTMModelInt8QAT(
        input_size=2, hidden_size=hidden, num_layers=layers, dropout_p=dropout,
        S_gate_q8=S_gate_q8, S_tanhc_q8=S_tanhc_q8
    ).to(device)  # :contentReference[oaicite:9]{index=9}
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=True)

    # 5) eval + metriche base
    mse_norm, y_pred, y_true, extras = _eval_epoch(model, dl_te, device, mu_y, std_y, collect_last_step=True)
    # mse_norm è calcolato su target normalizzati; ricalcoliamo tutto in °C per sicurezza
    def _mse(a,b): return float(np.mean((a-b)**2))
    def _mae(a,b): return float(np.mean(np.abs(a-b)))
    def _rmse(a,b): return _mse(a,b)**0.5
    def _r2(a,b):
        ss_res = np.sum((a-b)**2)
        ss_tot = np.sum((b-np.mean(b))**2) + 1e-12
        return float(1.0 - ss_res/ss_tot)
    MSE  = _mse(y_pred, y_true)
    MAE  = _mae(y_pred, y_true)
    RMSE = _rmse(y_pred, y_true)
    R2   = _r2(y_pred, y_true)

    print("\n=== TEST metrics (°C) ===")
    print(f"MSE : {MSE:.6f}")
    print(f"MAE : {MAE:.6f}")
    print(f"RMSE: {RMSE:.6f}")
    print(f"R²  : {R2:.4f}")

    # 6) Plot — parity & residual histogram
    import matplotlib.pyplot as plt

    # 6.1 Parity plot
    plt.figure()
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--", alpha=0.7, label="Ideal")
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    plt.xlabel("Ground truth Tjr (°C)")
    plt.ylabel("Predicted Tjr (°C)")
    plt.title(f"Parity plot — RMSE={RMSE:.3f} °C, R²={R2:.3f}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_parity.png", dpi=PLOT_DPI); plt.close()

    # 6.2 Residual histogram
    resid = y_pred - y_true
    plt.figure()
    plt.hist(resid, bins=60, alpha=0.9)
    plt.xlabel("Error = Pred − GT (°C)")
    plt.ylabel("Count")
    plt.title(f"Residuals histogram — mean={resid.mean():.3f}, std={resid.std():.3f}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_residual_hist.png", dpi=PLOT_DPI); plt.close()

    # 6.3 Error vs P/Tbp (usiamo l'ultimo step della finestra; de-normalizziamo)
    #    extras contiene P_last_norm e Tbp_last_norm. Convertiamoli in unità fisiche.
    if extras:
        P_last = extras["x_last_norm_P"] * std_x[0] + mu_x[0]
        Tbp_last = extras["x_last_norm_Tbp"] * std_x[1] + mu_x[1]
        plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(P_last, resid, s=4, alpha=0.5)
        plt.xlabel("P_last (W)")
        plt.ylabel("Error (°C)")
        plt.title("Error vs P_last")
        plt.grid(True)
        plt.subplot(2,1,2)
        plt.scatter(Tbp_last, resid, s=4, alpha=0.5)
        plt.xlabel("Tbp_last (°C)")
        plt.ylabel("Error (°C)")
        plt.title("Error vs Tbp_last")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "test_residual_vs_P_Tbp.png", dpi=PLOT_DPI); plt.close()

    # 7) Overlay temporale su alcuni cicli di test (originali)
    #    Per ciascun ciclo: predizione su tutte le finestre scorrevoli e confronto con GT e ODE.
    overlay_ids = idx_test[:max(0, int(args.overlay_cycles))]
    for k, i_cyc in enumerate(overlay_ids):
        t   = cycles_t[i_cyc]
        P   = cycles_P[i_cyc]
        Tbp = cycles_Tbp[i_cyc]
        Tjr = cycles_Tjr[i_cyc]

        # costruisci finestre e predici sequenza
        X = np.column_stack([P, Tbp])
        Xw, Yt = sliding_windows(X, Tjr)  # Xw: (Nw, W, 2), Yt: (Nw,)  :contentReference[oaicite:10]{index=10}
        # normalizza
        Xw_n = (Xw - mu_x) / std_x
        Xw_t = torch.from_numpy(Xw_n.astype(np.float32)).to(device)
        with torch.no_grad():
            y_hat_n = model(Xw_t).float().cpu().numpy()
        y_hat   = y_hat_n * std_y + mu_y
        t_eff   = t[WINDOW_SIZE - 1:]  # allineato alle finestre

        # baseline ODE su tutto il record
        T_ode, _ = solve_reference_ode(t, P, Tbp)  # :contentReference[oaicite:11]{index=11}

        # plot
        plt.figure(figsize=(10, 4))
        plt.plot(t, Tjr, label="GT Tjr", linewidth=1.2)
        plt.plot(t_eff, y_hat, label="LSTM pred", linewidth=1.2)
        plt.plot(t, T_ode, label="ODE baseline", linewidth=1.0, alpha=0.9)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Cycle #{i_cyc} — overlay (W={WINDOW_SIZE})")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"test_overlay_cycle_{k}.png", dpi=PLOT_DPI); plt.close()

    # 8) opzionale: salva CSV con tutte le predizioni/GT
    if args.save_csv:
        out_csv = Path(args.save_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd
        df = pd.DataFrame({"y_true_degC": y_true, "y_pred_degC": y_pred, "error_degC": (y_pred - y_true)})
        df.to_csv(out_csv, index=False)
        print(f"Saved full-window predictions to: {out_csv}")

if __name__ == "__main__":
    main()
