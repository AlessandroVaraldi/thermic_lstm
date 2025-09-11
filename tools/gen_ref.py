#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera una sequenza di riferimento (input Q15 + y_norm Q15) per confrontare l'inference C
contro la LSTM PyTorch quant-aware.

Output:
  - checkpoints/<name>.h     (header C con REF_W, REF_D, ref_x_q15[], ref_y_q15)
  - checkpoints/<name>.bin   (binario compatibile col main.c proposto)

Uso tipico:
  python -m tools.gen_ref --ckpt checkpoints/best_qat.pth --meta checkpoints/meta_qat.json --name ref_example
"""

from __future__ import annotations
import argparse, json, os, struct
from pathlib import Path

import numpy as np
import torch

# --- project-local (stesso schema di qat.py) ---
from src.config import (
    CSV_DIR, CSV_GLOB, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, SEED, AUG_CYCLES
)
from src.data_utils import seed_everything, load_all_csvs, compute_powers, augment_cycle
from src.dataset import WindowDataset
from src.qat_int8 import LSTMModelInt8QAT

CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- utils ----------------
def split_indices(n: int, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac   * n))
    return idx[:n_tr].tolist(), idx[n_tr:n_tr+n_va].tolist(), idx[n_tr+n_va:].tolist()

def to_q15(x: float) -> int:
    # clamp a ~[-1, 1) per sicurezza e converti in Q0.15
    xf = float(max(-0.999969, min(0.999969, x)))
    q = int(round(xf * (1<<15)))
    # saturazione esplicita
    if q >  32767: q =  32767
    if q < -32768: q = -32768
    return q

def dump_header(path: Path, name: str, x_q15: np.ndarray, y_q15: int):
    W, D = x_q15.shape
    lines = []
    lines.append("#pragma once")
    lines.append(f"#define REF_W {W}")
    lines.append(f"#define REF_D {D}")
    lines.append(f"static const int16_t ref_y_q15 = {int(y_q15)};")
    lines.append(f"static const int16_t ref_x_q15[{W*D}] = {{")
    flat = x_q15.reshape(-1).tolist()
    for i in range(0, len(flat), 16):
        chunk = ", ".join(str(int(v)) for v in flat[i:i+16])
        lines.append("  " + chunk + ("," if (i+16)<len(flat) else ""))
    lines.append("};")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ref] wrote {path}")

def dump_binary(path: Path, x_q15: np.ndarray, y_q15: int):
    W, D = x_q15.shape
    with open(path, "wb") as f:
        # header: magic 'REF1', W, D, y_q15
        f.write(struct.pack("<IHHh", 0x31464552, W, D, int(y_q15)))
        f.write(x_q15.astype(np.int16, copy=False).tobytes(order="C"))
    print(f"[ref] wrote {path}")

def build_cycles(augment: int, dt_hint: float | None = None):
    datasets = load_all_csvs()
    if len(datasets) == 0:
        raise FileNotFoundError(f"Nessun CSV in {CSV_DIR} (glob: {CSV_GLOB})")
    cycles_P, cycles_Tbp, cycles_Tjr = [], [], []
    # se serve, stima dt come mediana per uniformità (solo per eventuali shift di plotting)
    if dt_hint is None:
        dts = [float(np.median(np.diff(cols["t"]))) for cols in datasets]
        dt_hint = float(np.median(dts))
    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])
        cycles_P.append(P); cycles_Tbp.append(cols["Tbp"]); cycles_Tjr.append(cols["Tjr"])
        for _ in range(max(0, augment)):
            P2, Tbp2, Tjr2 = augment_cycle(P, cols["Tbp"], cols["Tjr"])
            cycles_P.append(P2); cycles_Tbp.append(Tbp2); cycles_Tjr.append(Tjr2)
    return cycles_P, cycles_Tbp, cycles_Tjr

def pick_one_sequence(cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y,
                      split="test", seed=SEED, idx_in_split=0):
    # Costruisci WindowDataset concatenato (come in train) per lo split richiesto.
    n = len(cycles_P)
    tr_idx, va_idx, te_idx = split_indices(n, seed=seed)
    pick = {"train": tr_idx, "val": va_idx, "test": te_idx}[split]
    if len(pick) == 0:
        # fallback: usa qualsiasi
        pick = list(range(n))
    # Concat sui cicli selezionati
    ds_list = []
    for i in pick:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y = cycles_Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    from torch.utils.data import ConcatDataset, DataLoader
    ds = ConcatDataset(ds_list)
    # DataLoader senza shuffle per avere un ordine stabile
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    # Estrai la sequenza idx_in_split-esima (o la prima)
    idx = max(0, min(idx_in_split, len(ds)-1))
    it = iter(dl)
    xb, yb = None, None
    for k, batch in enumerate(it):
        if k == idx:
            xb, yb = batch; break
    if xb is None:
        xb, yb = next(iter(dl))
    # xb: (1, W, D) già NORMALIZZATO come in training; yb: (1,)
    return xb[0].numpy(), float(yb[0].numpy())

def main():
    ap = argparse.ArgumentParser("Generate reference (PyTorch → Q15) per confronto con engine C")
    ap.add_argument("--ckpt", type=str, default=str(CKPT_DIR/"best_qat.pth"), help="checkpoint PyTorch")
    ap.add_argument("--meta", type=str, default=str(CKPT_DIR/"meta_qat.json"), help="meta JSON con hparams e stats (opzionale ma consigliato)")
    ap.add_argument("--name", type=str, default="ref_example", help="basename dei file di output")
    ap.add_argument("--outdir", type=str, default="ie/include", help="cartella di output per .h e .bin")
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--idx", type=int, default=0, help="indice della finestra nello split scelto")
    ap.add_argument("--augment", type=int, default=AUG_CYCLES, help="n. augment cycles (come in train)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--quant-eval", type=int, default=1, help="usa quant_eval(True) se disponibile")
    args = ap.parse_args()

    seed_everything(SEED)

    # ---- carica meta se presente ----
    meta_path = Path(args.meta)
    use_meta = meta_path.exists()
    if use_meta:
        meta = json.loads(meta_path.read_text())
        hidden   = int(meta.get("hidden", HIDDEN_SIZE))
        layers   = int(meta.get("layers", NUM_LAYERS))
        dropout  = float(meta.get("dropout", 0.0))
        S_gate_q8  = int(meta.get("S_gate_q8", 32))
        S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
        mu_x = np.array(meta["mu_x"], dtype=np.float32)
        std_x= np.array(meta["std_x"], dtype=np.float32)
        mu_y = float(meta["mu_y"]); std_y = float(meta["std_y"])
        dt_m = float(meta.get("dt_measured", 0.0))
        seed  = int(meta.get("seed", SEED))
    else:
        print("[warn] meta_qat.json non trovato: uso defaults da src.config")
        hidden, layers, dropout = HIDDEN_SIZE, NUM_LAYERS, 0.0
        S_gate_q8, S_tanhc_q8 = 32, 64
        # fallback: ricalcola stats dai CSV su train split
        cycles_P, cycles_Tbp, cycles_Tjr = build_cycles(args.augment)
        tr_idx, va_idx, te_idx = split_indices(len(cycles_P), seed=SEED)
        def cat_with_pad(arrs, pad_len=WINDOW_SIZE-1):
            pad = np.full(pad_len, np.nan, dtype=float)
            return np.concatenate([np.concatenate([a, pad]) for a in arrs])[:-pad_len]
        P_train   = cat_with_pad([cycles_P[i]   for i in tr_idx])
        Tbp_train = cat_with_pad([cycles_Tbp[i] for i in tr_idx])
        Tjr_train = cat_with_pad([cycles_Tjr[i] for i in tr_idx])
        mu_x  = np.nanmean(np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32)
        std_x = np.nanstd (np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32) + 1e-6
        mu_y  = float(np.nanmean(Tjr_train).astype(np.float32))
        std_y = float(np.nanstd (Tjr_train).astype(np.float32) + 1e-6)
        dt_m  = 0.0
        seed  = SEED

    # ---- ricostruisci il modello e carica pesi ----
    device = torch.device(args.device)
    model = LSTMModelInt8QAT(
        input_size=INPUT_SIZE,
        hidden_size=hidden,
        num_layers=layers,
        dropout_p=dropout,
        S_gate_q8=S_gate_q8,
        S_tanhc_q8=S_tanhc_q8,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- costruisci cicli e scegli una finestra ----
    cycles_P, cycles_Tbp, cycles_Tjr = build_cycles(args.augment, dt_hint=dt_m or None)
    xb_norm, y_norm_target = pick_one_sequence(
        cycles_P, cycles_Tbp, cycles_Tjr,
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        split=args.split, seed=seed, idx_in_split=args.idx
    )  # xb_norm: (W,D) già normalizzato

    # ---- predizione PyTorch ----
    xb_t = torch.from_numpy(xb_norm).unsqueeze(0).to(device)  # (1,W,D)
    if args.quant_eval and hasattr(model, "quant_eval"):
        cm = model.quant_eval(True)
    else:
        from contextlib import nullcontext; cm = nullcontext()
    with torch.no_grad(), cm:
        yhat_norm = model(xb_t).float().cpu().numpy().item()

    # ---- conversione a Q15 ----
    x_q15 = np.vectorize(to_q15, otypes=[np.int16])(xb_norm)
    y_ref_q15 = to_q15(yhat_norm)

    # ---- salvataggi ----
    base = Path(args.name).stem
    OUT_DIR = Path(args.outdir); OUT_DIR.mkdir(parents=True, exist_ok=True)
    hdr_path = OUT_DIR / f"{base}.h"
    bin_path = OUT_DIR / f"{base}.bin"
    dump_header(hdr_path, base, x_q15, y_ref_q15)
    dump_binary(bin_path, x_q15, y_ref_q15)

    # ---- report ----
    print("\n[ref] Dettagli")
    print(f"  W={x_q15.shape[0]}  D={x_q15.shape[1]}")
    print(f"  y_norm_fp={yhat_norm:.6f} -> y_q15={y_ref_q15}")
    print(f"  output: {hdr_path}  e  {bin_path}")

if __name__ == "__main__":
    main()

# python -u -m tools.gen_ref --outdir ie/include --name ref_val --split val --idx 3
