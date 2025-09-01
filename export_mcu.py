#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export INT8-QAT LSTM weights for MCU — compatible with qat_int8.py

Legge direttamente dallo state_dict:
  layers.{L}.ih.weight / .bias / .qx.scale / .qw.scale
  layers.{L}.hh.weight / .bias / .qx.scale / .qw.scale
  fc.weight, fc.bias

Usa meta_qat.json per: hidden, layers, mu_x/std_x, mu_y/std_y, S_gate_q8, S_tanhc_q8.
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

# ---------------- utils ----------------
def _round_clip_i8(a: np.ndarray) -> np.ndarray:
    y = np.rint(a).astype(np.int32)
    np.clip(y, -127, 127, out=y)
    return y.astype(np.int8)

def _load_meta(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(f"Meta JSON not found: {p}")
    return json.loads(p.read_text())

def _import_cfg():
    # (facoltativo) per estrarre WINDOW_SIZE, altrimenti lo lasciamo a -1
    try:
        from src.config import WINDOW_SIZE
        return int(WINDOW_SIZE)
    except Exception:
        try:
            from config import WINDOW_SIZE
            return int(WINDOW_SIZE)
        except Exception:
            return -1

def _match_layers(keys: List[str]) -> List[int]:
    """Trova gli indici layer presenti nello state_dict."""
    layers = set()
    pat = re.compile(r"^layers\.(\d+)\.(ih|hh)\.")
    for k in keys:
        m = pat.match(k)
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)

def _get(state: Dict[str, torch.Tensor], k: str) -> torch.Tensor:
    if k not in state:
        raise KeyError(f"Missing key: {k}")
    return state[k]

def _make_lut_sigmoid_q8(S_gate_q8: int) -> np.ndarray:
    # indice q in [-127..127] → x = q * (S/256) → y = sigmoid(x) in Q8
    q = torch.arange(-127, 128, dtype=torch.float32)
    x = q * (float(S_gate_q8) / 256.0)
    y = torch.sigmoid(x)
    return torch.round(y * 256.0).clamp(0, 256).to(torch.int16).cpu().numpy()

def _make_lut_tanh_q8(S_q8: int) -> np.ndarray:
    q = torch.arange(-127, 128, dtype=torch.float32)
    x = q * (float(S_q8) / 256.0)
    y = torch.tanh(x)
    return torch.round(y * 256.0).clamp(-256, 256).to(torch.int16).cpu().numpy()

# -------------- core --------------
@torch.no_grad()
def export_mcu(ckpt_path: Path, meta_path: Path, out_npz: Path,
               out_header: Path | None, shift_bits: int, verbose: bool):

    # 0) carica checkpoint
    raw = torch.load(ckpt_path, map_location="cpu")
    state: Dict[str, torch.Tensor] = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
    keys = list(state.keys())
    if verbose:
        print(f"[info] loaded {len(keys)} keys from {ckpt_path}")

    # 1) meta
    meta = _load_meta(meta_path)
    L = int(meta.get("layers", 1))
    H = int(meta.get("hidden"))
    S_gate_q8  = int(meta.get("S_gate_q8", 32))
    S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
    mu_x  = np.array(meta["mu_x"], dtype=np.float32)
    std_x = np.array(meta["std_x"], dtype=np.float32) + 1e-6
    mu_y  = np.float32(meta["mu_y"])
    std_y = np.float32(meta["std_y"])
    WIN   = _import_cfg()  # opzionale

    found_layers = _match_layers(keys)
    if not found_layers:
        raise RuntimeError("No layers.*.ih/hh keys found in state_dict (expected qat_int8.py naming).")
    if L != len(found_layers):
        print(f"[warn] meta says layers={L}, state_dict has layers={len(found_layers)} — user will rely on state_dict.")

    SHIFT = int(shift_bits)
    scale_to_q8 = 256.0 / float(S_gate_q8)

    # Per contenitori multi-layer
    W_ih_q8_list, W_hh_q8_list, b_q8_list = [], [], []
    M_ih_list, M_hh_list = [], []

    F_in_first = None

    for li in found_layers:
        # --- chiavi per layer li ---
        ih_w = f"layers.{li}.ih.weight"
        ih_b = f"layers.{li}.ih.bias"
        ih_qx= f"layers.{li}.ih.qx.scale"
        ih_qw= f"layers.{li}.ih.qw.scale"

        hh_w = f"layers.{li}.hh.weight"
        hh_b = f"layers.{li}.hh.bias"
        hh_qx= f"layers.{li}.hh.qx.scale"
        hh_qw= f"layers.{li}.hh.qw.scale"

        # --- tensori float ---
        W_ih_f = _get(state, ih_w).cpu().numpy()      # [4H, F]
        W_hh_f = _get(state, hh_w).cpu().numpy()      # [4H, H]
        b_ih_f = _get(state, ih_b).cpu().numpy()      # [4H]
        b_hh_f = _get(state, hh_b).cpu().numpy()      # [4H]

        # --- scale fake-quant (attivazioni & pesi) ---
        s_in_x = float(_get(state, ih_qx).item())
        s_w_ih = float(_get(state, ih_qw).item())
        s_in_h = float(_get(state, hh_qx).item())
        s_w_hh = float(_get(state, hh_qw).item())

        # --- quantizzazione pesi ---
        W_ih_q8 = _round_clip_i8(W_ih_f / s_w_ih)
        W_hh_q8 = _round_clip_i8(W_hh_f / s_w_hh)

        # --- bias totale (Q8, pre-attivazione) ---
        b_q8 = np.rint((b_ih_f + b_hh_f) * scale_to_q8).astype(np.int32)

        # --- moltiplicatori fixed-point ---
        M_ih = int(np.rint((s_in_x * s_w_ih) * scale_to_q8 * (1 << SHIFT)))
        M_hh = int(np.rint((s_in_h * s_w_hh) * scale_to_q8 * (1 << SHIFT)))

        W_ih_q8_list.append(W_ih_q8)
        W_hh_q8_list.append(W_hh_q8)
        b_q8_list.append(b_q8)
        M_ih_list.append(M_ih)
        M_hh_list.append(M_hh)

        if li == 0:
            F_in_first = W_ih_f.shape[1]

        if verbose:
            print(f"[L{li}] ih: s_in={s_in_x:.6g}, s_w={s_w_ih:.6g}  |  hh: s_in={s_in_h:.6g}, s_w={s_w_hh:.6g}")

    # 2) FC finale (float32)
    fc_w = _get(state, "fc.weight").cpu().numpy().astype(np.float32)  # [1, 2H]
    fc_b = _get(state, "fc.bias").cpu().numpy().astype(np.float32)    # [1]

    # 3) LUT per attivazioni (deploy)
    lut_sig_gate  = _make_lut_sigmoid_q8(S_gate_q8)
    lut_tanh_gate = _make_lut_tanh_q8   (S_gate_q8)
    lut_tanh_c    = _make_lut_tanh_q8   (S_tanhc_q8)

    qx_ih_scales, qx_hh_scales = [], []

    for li in found_layers:
        ih_qx = float(state[f"layers.{li}.ih.qx.scale"].item())
        ih_qw = float(state[f"layers.{li}.ih.qw.scale"].item())
        hh_qx = float(state[f"layers.{li}.hh.qx.scale"].item())
        hh_qw = float(state[f"layers.{li}.hh.qw.scale"].item())
        # (… usa ih_qw/hh_qw per quantizzare i pesi come già fai …)
        qx_ih_scales.append(ih_qx)
        qx_hh_scales.append(hh_qx)

    # 4) salva .npz
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        num_layers=np.int32(len(found_layers)),
        input_size=np.int32(int(F_in_first)),
        hidden_size=np.int32(H),
        window_size=np.int32(WIN),
        S_gate_q8=np.int32(S_gate_q8),
        S_tanhc_q8=np.int32(S_tanhc_q8),
        shift=np.int32(SHIFT),
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        ih_w_q8=np.stack(W_ih_q8_list, axis=0),   # [L, 4H, F]
        hh_w_q8=np.stack(W_hh_q8_list, axis=0),   # [L, 4H, H]
        b_q8=np.stack(b_q8_list, axis=0),         # [L, 4H]
        M_ih=np.array(M_ih_list, dtype=np.int32), # [L]
        M_hh=np.array(M_hh_list, dtype=np.int32), # [L]
        lut_sig_gate=lut_sig_gate.astype(np.int16),
        lut_tanh_gate=lut_tanh_gate.astype(np.int16),
        lut_tanh_c=lut_tanh_c.astype(np.int16),
        fc_w=fc_w, fc_b=fc_b,
        QX_IH_SCALES=np.array(qx_ih_scales, dtype=np.float32),
        QX_HH_SCALES=np.array(qx_hh_scales, dtype=np.float32),
        c_clip=np.float32(meta.get("c_clip", 8.0))
    )
    print(f"[OK] saved {out_npz}")

    # 5) opzionale: header C
    if out_header:
        def arr(name, a, ctype):
            flat = a.reshape(-1)
            return f"static const {ctype} {name}[{flat.size}] = {{ " + ", ".join(map(str, flat.tolist())) + " };\n"
        Hh, Ff, Ln = int(H), int(F_in_first), int(len(found_layers))

        hdr  = []
        hdr += ["/** Auto-generated by export_mcu.py **/\n#include <stdint.h>\n\n"]
        hdr += [f"#define LSTM_L   {Ln}\n#define LSTM_F   {Ff}\n#define LSTM_H   {Hh}\n#define LSTM_4H  {4*Hh}\n"]
        hdr += [f"#define WIN_SIZE {WIN}\n\n"]
        hdr += [f"#define S_GATE_Q8   {S_gate_q8}\n#define S_TANHC_Q8  {S_tanhc_q8}\n"]
        hdr += [f"#define FX_SHIFT   {SHIFT}\n"]
        hdr += [f"#define C_CLIP {float(meta.get('c_clip', 8.0)):.8f}f\n"]
        hdr += [arr("FX_M_IH", np.array(M_ih_list, dtype=np.int32), "int32_t")]
        hdr += [arr("FX_M_HH", np.array(M_hh_list, dtype=np.int32), "int32_t")]
        hdr += [f"static const float MU_X[2]={{ {mu_x[0]:.8f}f, {mu_x[1]:.8f}f }};\n"]
        hdr += [f"static const float STD_X[2]={{ {std_x[0]:.8f}f, {std_x[1]:.8f}f }};\n"]
        hdr += [f"static const float MU_Y={mu_y:.8f}f;\nstatic const float STD_Y={std_y:.8f}f;\n\n"]
        hdr += [arr("IH_W_Q8", np.stack(W_ih_q8_list, axis=0).astype(np.int8), "int8_t")]
        hdr += [arr("HH_W_Q8", np.stack(W_hh_q8_list, axis=0).astype(np.int8), "int8_t")]
        hdr += [arr("B_Q8",    np.stack(b_q8_list,     axis=0).astype(np.int32), "int32_t")]
        hdr += [arr("LUT_SIG_GATE_Q8",  lut_sig_gate.astype(np.int16),  "int16_t")]
        hdr += [arr("LUT_TANH_GATE_Q8", lut_tanh_gate.astype(np.int16), "int16_t")]
        hdr += [arr("LUT_TANH_C_Q8",    lut_tanh_c.astype(np.int16),    "int16_t")]
        hdr += [arr("FC_W", fc_w.astype(np.float32), "float")]
        hdr += [arr("FC_B", fc_b.astype(np.float32), "float")]

        def arrf(name, a):
            flat = np.asarray(a, dtype=np.float32).reshape(-1)
            items = ", ".join([f"{x:.8f}f" for x in flat.tolist()])
            return f"static const float {name}[{flat.size}] = {{ {items} }};\n"

        hdr += [arrf("QX_IH_SCALES", qx_ih_scales)]
        hdr += [arrf("QX_HH_SCALES", qx_hh_scales)]

        out_header.parent.mkdir(parents=True, exist_ok=True)
        out_header.write_text("".join(hdr))
        print(f"[OK] saved {out_header}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   type=str, default="checkpoints/best_qat.pth")
    ap.add_argument("--meta",   type=str, default="checkpoints/meta_qat.json")
    ap.add_argument("--out",    type=str, default="checkpoints/mcu_export.npz")
    ap.add_argument("--header", type=str, default="checkpoints/model_int8.h")
    ap.add_argument("--shift",  type=int, default=24, help="bit-shift fixed-point (tipico 24)")
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    export_mcu(
        ckpt_path=Path(args.ckpt),
        meta_path=Path(args.meta),
        out_npz=Path(args.out),
        out_header=None if args.no_header else Path(args.header),
        shift_bits=int(args.shift),
        verbose=bool(args.verbose),
    )

if __name__ == "__main__":
    main()
