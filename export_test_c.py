#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esporta il TEST set (stesso split di qlstm_train.py) in un header C.

Output: un .h con
  #define TEST_N ...
  #define WIN_SIZE ...
  #define LSTM_F 2
  static const float  TEST_X[TEST_N][WIN_SIZE][2];
  static const float  TEST_Y[TEST_N];
  // opzionali:
  static const int8_t TEST_XQ[TEST_N][WIN_SIZE][2];   // se --quantize e --ckpt
  static const float  TEST_X_NORM[TEST_N][WIN_SIZE][2];
  static const float  TEST_Y_NORM[TEST_N];

Dipendenze: legge CSV via src.data_utils/load_all_csvs e compute_powers (P da Id/Iq),
costruisce finestre come in training (sliding_windows, target a fine finestra).
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

# --- project-local (stessi import del training)
from src.config import WINDOW_SIZE, CSV_DIR, CSV_GLOB
from src.data_utils import load_all_csvs, compute_powers, sliding_windows   # features & windowing
# NOTE: non importiamo WindowDataset per evitare normalizzazione on-the-fly:
# qui esportiamo finestre RAW (più eventuale normalizzazione/quantizzazione off-line).

def _build_original_cycles():
    """Come in qtest.py: SOLO cicli originali (no augmentation)."""
    datasets = load_all_csvs()
    if len(datasets) == 0:
        raise FileNotFoundError(f"Nessun CSV in {CSV_DIR} (glob: {CSV_GLOB})")
    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []
    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])
        cycles_t.append(cols["t"])
        cycles_P.append(P)
        cycles_Tbp.append(cols["Tbp"])
        cycles_Tjr.append(cols["Tjr"])
    return cycles_t, cycles_P, cycles_Tbp, cycles_Tjr

def _windows_from_cycle(P: np.ndarray, Tbp: np.ndarray, Tjr: np.ndarray):
    """
    Genera finestre (RAW, no normalizzazione) come in training:
      X = [P, Tbp] con target Tjr allineato all'ULTIMO passo finestra.
    Applica mask di finitezza come WindowDataset.
    """
    X = np.column_stack([P, Tbp])  # ordine delle feature come in build_concat_dataset
    Xw, yw = sliding_windows(X, Tjr)  # shape -> (Nw, W, 2), (Nw,)
    mask = np.isfinite(Xw).all(axis=(1, 2)) & np.isfinite(yw)
    return Xw[mask], yw[mask]

def _fmt_floats(a: np.ndarray, ndigits: int = 6) -> str:
    flat = a.reshape(-1).astype(np.float32)
    fmt = f"{{:.{ndigits}f}}f"          # es: 1.000000f, 2.500000f
    return ", ".join(fmt.format(float(x)) for x in flat)

def _fmt_int8(a: np.ndarray) -> str:
    flat = a.reshape(-1).astype(np.int8)
    return ", ".join(str(int(x)) for x in flat.tolist())

def main():
    ap = argparse.ArgumentParser("Export TEST set → C header")
    ap.add_argument("--meta",   type=str, default="checkpoints/meta_qat.json")
    ap.add_argument("--out-h",  type=str, default="checkpoints/test_data.h")
    ap.add_argument("--ckpt",   type=str, default="", help="(opzionale) checkpoint per leggere la scala qx del primo layer per --quantize")
    ap.add_argument("--quantize", action="store_true", help="esporta anche TEST_XQ quantizzato int8 come nel 1° layer")
    ap.add_argument("--also-norm", action="store_true", help="esporta anche versioni normalizzate TEST_X_NORM/TEST_Y_NORM")
    ap.add_argument("--max-n",  type=int, default=0, help="limita a N finestre (0 = tutte)")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    mu_x  = np.array(meta["mu_x"], dtype=np.float32)
    std_x = np.array(meta["std_x"], dtype=np.float32) + 1e-6
    mu_y  = float(meta["mu_y"])
    std_y = float(meta["std_y"])

    # split test salvato in training
    idx_test_raw = list(map(int, meta["splits"]["test"]))

    # cicli originali (no augmentation) e filtro come qtest.py
    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = _build_original_cycles()
    n_original = len(cycles_P)
    idx_test = [i for i in idx_test_raw if i < n_original]

    # costruisci finestre RAW per ogni ciclo test
    X_list, Y_list = [], []
    for i in idx_test:
        Xw, yw = _windows_from_cycle(cycles_P[i], cycles_Tbp[i], cycles_Tjr[i])
        X_list.append(Xw)   # (Ni, W, 2)
        Y_list.append(yw)   # (Ni,)
    if not X_list:
        raise RuntimeError("Split di test vuoto dopo il filtro 'no augmentation'.")

    X_all = np.concatenate(X_list, axis=0).astype(np.float32)  # (N, W, 2) in unità fisiche
    Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)  # (N,)

    if args.max_n and args.max_n > 0:
        N = min(int(args.max_n), X_all.shape[0])
        X_all = X_all[:N]
        Y_all = Y_all[:N]

    # opzionale: normalizzato come durante l'inferenza PyTorch
    if args.also_norm:
        X_norm = (X_all - mu_x.reshape(1, 1, -1)) / std_x.reshape(1, 1, -1)
        Y_norm = (Y_all - mu_y) / std_y

    # opzionale: quantizzato int8 come il PRIMO layer (serve scala qx)
    if args.quantize:
        assert args.ckpt, "--quantize richiede --ckpt"
        state = torch.load(args.ckpt, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and "model" in state else state
        try:
            s_qx_ih0 = float(sd["layers.0.ih.qx.scale"].item())
        except KeyError as e:
            raise KeyError("Manca 'layers.0.ih.qx.scale' nel checkpoint: serve il modello QAT.") from e
        # quantizza le caratteristiche NORMALIZZATE come fa l'engine prima del 1° layer
        Xn = (X_all - mu_x.reshape(1, 1, -1)) / std_x.reshape(1, 1, -1)
        Xq = np.rint(Xn / s_qx_ih0).astype(np.int32)
        Xq = np.clip(Xq, -127, 127).astype(np.int8)

    # --- scrivi header C ---
    out_h = Path(args.out_h)
    out_h.parent.mkdir(parents=True, exist_ok=True)

    N = int(X_all.shape[0]); W = int(X_all.shape[1]); F = int(X_all.shape[2])
    assert W == int(WINDOW_SIZE) and F == 2, "Shape inattesa (W o F). Controlla WINDOW_SIZE/feature order."

    lines = []
    lines += ["/** Auto-generated TEST dataset — from export_test_c.py **/\n"]
    lines += ["#ifndef TEST_DATA_H\n#define TEST_DATA_H\n\n#include <stdint.h>\n\n"]
    lines += [f"#define TEST_N   {N}\n#define WIN_SIZE {W}\n#define LSTM_F   {F}\n\n"]

    # statistiche (comode in C)
    lines += [f"static const float MU_X[2]  = {{ {mu_x[0]:.8f}f, {mu_x[1]:.8f}f }};\n"]
    lines += [f"static const float STD_X[2] = {{ {std_x[0]:.8f}f, {std_x[1]:.8f}f }};\n"]
    lines += [f"static const float MU_Y = {mu_y:.8f}f;\nstatic const float STD_Y = {std_y:.8f}f;\n\n"]

    # X e Y (RAW, float)
    lines += [f"static const float TEST_X[{N}][{W}][{F}] = {{\n"]
    for n in range(N):
        lines += ["  {\n"]
        for t in range(W):
            lines += [f"    {{ {_fmt_floats(X_all[n, t, :])} }},\n"]
        lines += ["  },\n"]
    lines += ["};\n\n"]

    lines += [f"static const float TEST_Y[{N}] = {{ {_fmt_floats(Y_all)} }};\n\n"]

    # opzionali
    if args.also_norm:
        lines += [f"static const float TEST_X_NORM[{N}][{W}][{F}] = {{\n"]
        for n in range(N):
            lines += ["  {\n"]
            for t in range(W):
                lines += [f"    {{ {_fmt_floats(X_norm[n, t, :])} }},\n"]
            lines += ["  },\n"]
        lines += ["};\n\n"]
        lines += [f"static const float TEST_Y_NORM[{N}] = {{ {_fmt_floats(Y_norm)} }};\n\n"]

    if args.quantize:
        lines += [f"static const int8_t TEST_XQ[{N}][{W}][{F}] = {{\n"]
        for n in range(N):
            lines += ["  {\n"]
            for t in range(W):
                lines += [f"    {{ {_fmt_int8(Xq[n, t, :])} }},\n"]
            lines += ["  },\n"]
        lines += ["};\n\n"]

    lines += ["#endif // TEST_DATA_H\n"]

    out_h.write_text("".join(lines), encoding="utf-8")
    print(f"[OK] scritto {out_h}  (finestre: {N}, W={W}, F={F})")

if __name__ == "__main__":
    main()
