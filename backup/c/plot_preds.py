#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legge i file binari prodotti da main.c (float32, nativo) e plottail confronto
tra y_true e y_pred. Stampa MAE/RMSE, salva una figura PNG.

Uso:
  python plot_preds.py --preds preds.bin --truth truth.bin --out plot.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="file binario float32 con le predizioni")
    ap.add_argument("--truth", required=True, help="file binario float32 con la ground truth")
    ap.add_argument("--out",   default="pred_vs_truth.png")
    ap.add_argument("--endianness", choices=["native", "little", "big"], default="native",
                    help="interpretazione endianness dei binari (default: native)")
    args = ap.parse_args()

    if args.endianness == "native":
        dt = np.dtype("float32")
    elif args.endianness == "little":
        dt = np.dtype("<f4")
    else:
        dt = np.dtype(">f4")

    y_pred = np.fromfile(args.preds, dtype=dt)
    y_true = np.fromfile(args.truth, dtype=dt)

    n = min(y_pred.size, y_true.size)
    if y_pred.size != y_true.size:
        print(f"[warn] dimensioni diverse: preds={y_pred.size}, truth={y_true.size}. Uso n={n}")
    y_pred = y_pred[:n]
    y_true = y_true[:n]

    mae  = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    print(f"N={n}  MAE={mae:.6f}  RMSE={rmse:.6f}")

    # Figura: serie + scatter diagonale
    fig = plt.figure(figsize=(11,5))

    # subplot 1: serie nel tempo
    ax1 = fig.add_subplot(1,2,1)
    ax1.plot(y_true, label="y_true")
    ax1.plot(y_pred, label="y_pred", alpha=0.8)
    ax1.set_title(f"Serie — N={n}\nMAE={mae:.3f}  RMSE={rmse:.3f}")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("Tjr")
    ax1.legend(loc="best")

    # subplot 2: scatter pred vs true + diagonale
    ax2 = fig.add_subplot(1,2,2)
    ax2.scatter(y_true, y_pred, s=8, alpha=0.6)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax2.plot([lo, hi], [lo, hi], linestyle="--")
    ax2.set_title("Pred vs True")
    ax2.set_xlabel("True")
    ax2.set_ylabel("Pred")
    ax2.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[OK] salvato {args.out}")

if __name__ == "__main__":
    main()

# python plot_preds.py --preds out_preds.bin --truth out_truth.bin --out plot.png