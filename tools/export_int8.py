#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-shot pipeline:
  1) Export INT8 bin+json from best checkpoint (meta > quant_meta > infer)
  2) Emit C header with offsets/shapes (model_offsets.h)
  3) Generate Q15 reference pair (.h + .bin) from the QAT model
Relies on src.config for almost everything; tweak constants below if needed.
"""

from __future__ import annotations
import json, re, struct
from pathlib import Path
from typing import List
import numpy as np
import torch

# ---- project-local ----
from src.config import (
    CSV_DIR, CSV_GLOB, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, SEED, AUG_CYCLES,
    DEVICE, BATCH_SIZE
)
from src.data_utils import seed_everything, load_all_csvs, compute_powers, augment_cycle
from src.dataset import WindowDataset
from src.qat_int8 import LSTMModelInt8QAT

# ---- local knobs (keep minimal; no CLI) ----
CKPT_PATH        = Path("checkpoints/best_qat.pth")
META_PATH        = Path("checkpoints/meta_qat.json")
QUANT_META_PATH  = Path("checkpoints/quant_meta.json")
OUT_CHECKPOINTS  = Path("checkpoints")        # model_int8.bin/json
OUT_INCLUDE      = Path("ie/include")         # headers and ref files
REF_NAME         = "ref_auto"                 # base name for ref header/bin
CAL_MAX_BATCHES  = 8                          # batches for pre-act calibration
REF_SPLIT        = "val"                      # "train"|"val"|"test"
REF_IDX          = 0                          # window index in chosen split

# ---------------- Shared utilities ----------------
def split_indices(n: int, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    """Shuffle and split indices into train/val/test (cycle-level)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac   * n))
    return idx[:n_tr].tolist(), idx[n_tr:n_tr+n_va].tolist(), idx[n_tr+n_va:].tolist()

def build_cycles(augment: int):
    """Load CSVs and optionally augment cycles."""
    datasets = load_all_csvs()
    if len(datasets) == 0:
        raise FileNotFoundError(f"No CSVs in {CSV_DIR} (glob: {CSV_GLOB})")
    cycles_P, cycles_Tbp, cycles_Tjr = [], [], []
    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])
        cycles_P.append(P); cycles_Tbp.append(cols["Tbp"]); cycles_Tjr.append(cols["Tjr"])
        for _ in range(max(0, augment)):
            P2, Tbp2, Tjr2 = augment_cycle(P, cols["Tbp"], cols["Tjr"])
            cycles_P.append(P2); cycles_Tbp.append(Tbp2); cycles_Tjr.append(Tjr2)
    return cycles_P, cycles_Tbp, cycles_Tjr

def compute_stats_from_train(cycles_P, cycles_Tbp, cycles_Tjr, seed=SEED):
    """Train-split mean/std for inputs/target."""
    tr, _, _ = split_indices(len(cycles_P), seed=seed)
    PAD = WINDOW_SIZE - 1
    def cat_with_pad(arrs: List[np.ndarray]):
        pad = np.full(PAD, np.nan, dtype=float)
        return np.concatenate([np.concatenate([a, pad]) for a in arrs])[:-PAD]
    P_train   = cat_with_pad([cycles_P[i]   for i in tr])
    Tbp_train = cat_with_pad([cycles_Tbp[i] for i in tr])
    Tjr_train = cat_with_pad([cycles_Tjr[i] for i in tr])
    mu_x  = np.nanmean(np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32)
    std_x = np.nanstd (np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32) + 1e-6
    mu_y  = float(np.nanmean(Tjr_train).astype(np.float32))
    std_y = float(np.nanstd (Tjr_train).astype(np.float32) + 1e-6)
    return (mu_x, std_x, mu_y, std_y)

def infer_dims_from_state(state_dict):
    """Infer input_size, H, num_layers from state_dict."""
    ih_keys = [k for k in state_dict.keys() if re.search(r"^layers\.\d+\.ih\.weight$", k)]
    if not ih_keys:
        raise RuntimeError("Cannot infer dims: ih.weight not found")
    k0 = sorted(ih_keys)[0]
    w = state_dict[k0]
    out4H, in_features = w.shape
    H = int(out4H // 4)
    input_size = int(in_features)
    layer_idx = set(int(re.findall(r"^layers\.(\d+)\.", k)[0]) for k in state_dict.keys() if k.startswith("layers."))
    num_layers = int(max(layer_idx) + 1) if layer_idx else 1
    return input_size, H, num_layers

def to_q15(x: float) -> int:
    """Float -> Q0.15 with clamp."""
    xf = float(max(-0.999969, min(0.999969, x)))
    q = int(round(xf * (1<<15)))
    if q >  32767: q =  32767
    if q < -32768: q = -32768
    return q

def dump_header_ref(path: Path, x_q15: np.ndarray, y_q15: int):
    """Write C header with reference vectors."""
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

def dump_binary_ref(path: Path, x_q15: np.ndarray, y_q15: int):
    """Write binary blob for C test harness."""
    W, D = x_q15.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHh", 0x31464552, W, D, int(y_q15)))  # 'REF1'
        f.write(x_q15.astype(np.int16, copy=False).tobytes(order="C"))
    print(f"[ref] wrote {path}")

def pick_one_sequence(cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y,
                      split="test", seed=SEED, idx_in_split=0):
    """Return (W,D) normalized window and target from the chosen split."""
    tr_idx, va_idx, te_idx = split_indices(len(cycles_P), seed=seed)
    pick = {"train": tr_idx, "val": va_idx, "test": te_idx}[split]
    if len(pick) == 0:
        pick = list(range(len(cycles_P)))
    ds_list = []
    for i in pick:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y = cycles_Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    from torch.utils.data import ConcatDataset, DataLoader
    ds = ConcatDataset(ds_list)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    idx = max(0, min(idx_in_split, len(ds)-1))
    it = iter(dl)
    xb, yb = None, None
    for k, batch in enumerate(it):
        if k == idx:
            xb, yb = batch; break
    if xb is None:
        xb, yb = next(iter(dl))
    return xb[0].numpy(), float(yb[0].numpy())

# ---------------- Main pipeline ----------------
def main():
    seed_everything(SEED)
    device = torch.device(DEVICE)
    OUT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    OUT_INCLUDE.mkdir(parents=True, exist_ok=True)

    # --- load checkpoint ---
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # --- choose meta: meta_qat.json -> quant_meta.json -> infer ---
    meta, use = {}, "infer"
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text()); use = "meta_qat"
        print(f"[meta] using {META_PATH}")
    elif QUANT_META_PATH.exists():
        meta = json.loads(QUANT_META_PATH.read_text()); use = "quant_meta"
        print(f"[meta] using {QUANT_META_PATH}")
    else:
        print("[meta] none found: inferring from checkpoint + CSV stats")

    # --- stats + params ---
    if use == "meta_qat":
        hidden   = int(meta.get("hidden", HIDDEN_SIZE))
        layers   = int(meta.get("layers", NUM_LAYERS))
        dropout  = float(meta.get("dropout", 0.0))
        S_gate_q8  = int(meta.get("S_gate_q8", 32))
        S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
        mu_x = np.array(meta["mu_x"], dtype=np.float32)
        std_x= np.array(meta["std_x"], dtype=np.float32)
        mu_y = float(meta["mu_y"]); std_y = float(meta["std_y"])
        input_size = INPUT_SIZE
    elif use == "quant_meta":
        hidden   = int(meta.get("hidden", HIDDEN_SIZE))
        layers   = int(meta.get("layers", NUM_LAYERS))
        dropout  = float(meta.get("dropout", 0.0))
        S_gate_q8  = int(meta.get("S_gate_q8", 32))
        S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
        if "norm" in meta:
            mu_x = np.array(meta["norm"]["mu_x"], dtype=np.float32)
            std_x= np.array(meta["norm"]["std_x"], dtype=np.float32)
            mu_y = float(meta["norm"]["mu_y"]); std_y = float(meta["norm"]["std_y"])
        else:
            cycles_P, cycles_Tbp, cycles_Tjr = build_cycles(AUG_CYCLES)
            mu_x, std_x, mu_y, std_y = compute_stats_from_train(cycles_P, cycles_Tbp, cycles_Tjr)
        input_size = int(meta.get("input_size", INPUT_SIZE))
    else:
        input_size, hidden, layers = infer_dims_from_state(state)
        dropout = 0.0
        S_gate_q8, S_tanhc_q8 = 32, 64
        cycles_P, cycles_Tbp, cycles_Tjr = build_cycles(AUG_CYCLES)
        mu_x, std_x, mu_y, std_y = compute_stats_from_train(cycles_P, cycles_Tbp, cycles_Tjr)

    # --- build model ---
    model = LSTMModelInt8QAT(
        input_size=input_size, hidden_size=hidden, num_layers=layers,
        dropout_p=dropout, S_gate_q8=S_gate_q8, S_tanhc_q8=S_tanhc_q8
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # --- calib dataset (val -> train) ---
    if 'cycles_P' not in locals():
        cycles_P, cycles_Tbp, cycles_Tjr = build_cycles(AUG_CYCLES)
    tr_idx, va_idx, _ = split_indices(len(cycles_P))
    def _build_concat(idxs):
        from torch.utils.data import ConcatDataset
        ds_list = []
        for i in idxs:
            X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
            y = cycles_Tjr[i]
            ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
        return ConcatDataset(ds_list) if ds_list else None
    ds_va = _build_concat(va_idx)
    ds_tr = _build_concat(tr_idx)
    cal_ds = ds_va if (ds_va is not None and len(ds_va) > 0) else ds_tr
    from torch.utils.data import DataLoader
    cal_loader = DataLoader(cal_ds, batch_size=max(1, BATCH_SIZE), shuffle=False,
                            num_workers=0, pin_memory=False)

    # --- pre-act calibration ---
    if not hasattr(model, "calibrate_preact_scales"):
        raise RuntimeError("model.calibrate_preact_scales() missing")
    pre_scales = model.calibrate_preact_scales(cal_loader, device, max_batches=CAL_MAX_BATCHES)
    for li in range(len(pre_scales)):
        for k in ("i","f","g","o"):
            v = float(pre_scales[li][k])
            pre_scales[li][k] = (1e-8 if (not np.isfinite(v) or v <= 0.0) else v)

    # --- quantize + write bin/json ---
    OUT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    bin_path  = OUT_CHECKPOINTS / "model_int8.bin"
    json_path = OUT_CHECKPOINTS / "model_int8.json"
    offsets = {"layers": [], "fc": {}}
    pkg = {
        "format": "INT8-LSTM-v1",
        "layers": [],
        "fc": {},
        "pre_scales": pre_scales,
        "norm": {"mu_x": mu_x.tolist(), "std_x": std_x.tolist(), "mu_y": mu_y, "std_y": std_y},
        "hidden": int(hidden), "layers_n": int(layers), "input_size": int(input_size)
    }

    import numpy as _np
    off_bytes = 0
    def _write_arr(fh, tensor, dtype):
        nonlocal off_bytes
        arr = tensor.detach().cpu().numpy().astype(dtype, copy=False)
        if arr.dtype == _np.int32 and arr.dtype.byteorder not in ('<','='):
            arr = arr.newbyteorder('<')  # little-endian for i32
        data = arr.tobytes(order="C")
        start = off_bytes
        fh.write(data)
        off_bytes += len(data)
        return start, len(data)

    with open(bin_path.with_suffix(".bin.tmp"), "wb") as fh:
        for li, cell in enumerate(model.layers):
            layer_entry = {"idx": li, "ih": {}, "hh": {},
                           "S_gate_q8": int(cell.sigmoid_q8.S_gate_q8),
                           "S_tanhc_q8": int(cell.tanh_q8_c.S_tanhc_q8),
                           "pre_scales": pre_scales[li]}
            offsets["layers"].append({"idx": li})

            if not hasattr(model, "quantize_linear_int8") or not hasattr(model, "compute_requant"):
                raise RuntimeError("quantize_linear_int8/compute_requant missing")

            # IH
            ih_W, ih_b, ih_Sx, ih_Sw = model.quantize_linear_int8(cell.ih)
            layer_entry["ih"]["W_shape"] = list(ih_W.shape)
            w_off, w_nbytes = _write_arr(fh, ih_W, _np.int8)
            offsets["layers"][-1]["ih_W_off"] = int(w_off); offsets["layers"][-1]["ih_W_nbytes"] = int(w_nbytes)
            if cell.ih.bias is not None:
                b_off, b_nbytes = _write_arr(fh, ih_b, _np.int32)
                offsets["layers"][-1]["ih_b_off"] = int(b_off); offsets["layers"][-1]["ih_b_nbytes"] = int(b_nbytes)
            layer_entry["ih"]["Sx"] = float(ih_Sx); layer_entry["ih"]["Sw"] = float(ih_Sw)
            rq_ih = {}
            for k in ("i","f","g","o"):
                m, r = model.compute_requant(ih_Sx, ih_Sw, pre_scales[li][k])
                rq_ih[k] = {"mult_q15": int(m), "rshift": int(r)}
            layer_entry["ih"]["requant"] = rq_ih

            # HH
            hh_W, hh_b, hh_Sx, hh_Sw = model.quantize_linear_int8(cell.hh)
            layer_entry["hh"]["W_shape"] = list(hh_W.shape)
            w_off, w_nbytes = _write_arr(fh, hh_W, _np.int8)
            offsets["layers"][-1]["hh_W_off"] = int(w_off); offsets["layers"][-1]["hh_W_nbytes"] = int(w_nbytes)
            if cell.hh.bias is not None:
                b_off, b_nbytes = _write_arr(fh, hh_b, _np.int32)
                offsets["layers"][-1]["hh_b_off"] = int(b_off); offsets["layers"][-1]["hh_b_nbytes"] = int(b_nbytes)
            layer_entry["hh"]["Sx"] = float(hh_Sx); layer_entry["hh"]["Sw"] = float(hh_Sw)
            rq_hh = {}
            for k in ("i","f","g","o"):
                m, r = model.compute_requant(hh_Sx, hh_Sw, pre_scales[li][k])
                rq_hh[k] = {"mult_q15": int(m), "rshift": int(r)}
            layer_entry["hh"]["requant"] = rq_hh

            pkg["layers"].append(layer_entry)

        # FC head
        fc_W, fc_b, fc_Sx, fc_Sw = model.quantize_linear_int8(model.fc)
        pkg["fc"]["W_shape"] = list(fc_W.shape)  # [1,2H]
        w_off, w_nbytes = _write_arr(fh, fc_W, _np.int8)
        offsets["fc"]["W_off"] = int(w_off); offsets["fc"]["W_nbytes"] = int(w_nbytes)
        if model.fc.bias is not None:
            b_off, b_nbytes = _write_arr(fh, fc_b, _np.int32)
            offsets["fc"]["b_off"] = int(b_off); offsets["fc"]["b_nbytes"] = int(b_nbytes)
        pkg["fc"]["Sx"] = float(fc_Sx); pkg["fc"]["Sw"] = float(fc_Sw)

    # atomic writes
    bin_tmp = bin_path.with_suffix(".bin.tmp"); bin_tmp.replace(bin_path)
    pkg["offsets"] = offsets
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(pkg, indent=2))
    json_tmp.replace(json_path)
    print(f"[export int8] saved: {bin_path}, {json_path}")

    # --- model_offsets.h directly from pkg/offsets (no re-read) ---
    mo_lines = []
    W = mo_lines.append
    L = len(pkg["layers"])
    W("// Auto-generated from model_int8.json")
    W("#pragma once")
    W("#include <stdint.h>")
    W("")
    W(f"#define MODEL_LAYERS {L}")
    W(f"#define MODEL_HIDDEN ( (int){ pkg['layers'][0]['ih']['W_shape'][0] // 4 } )")
    W("")
    for li, layer in enumerate(pkg["layers"]):
        o = pkg["offsets"]["layers"][li]
        ihs = layer["ih"]["W_shape"]; hhs = layer["hh"]["W_shape"]
        W(f"// Layer {li}")
        W(f"#define L{li}_IH_W_OFF     {o['ih_W_off']}u")
        W(f"#define L{li}_IH_W_NBYTES  {o['ih_W_nbytes']}u")
        if 'ih_b_off' in o:
            W(f"#define L{li}_IH_B_OFF     {o['ih_b_off']}u")
            W(f"#define L{li}_IH_B_NBYTES  {o['ih_b_nbytes']}u")
        else:
            W(f"#define L{li}_IH_B_OFF     0u")
            W(f"#define L{li}_IH_B_NBYTES  0u")
        W(f"#define L{li}_IH_OUT        {ihs[0]}")
        W(f"#define L{li}_IH_IN         {ihs[1]}")
        W(f"#define L{li}_HH_W_OFF     {o['hh_W_off']}u")
        W(f"#define L{li}_HH_W_NBYTES  {o['hh_W_nbytes']}u")
        if 'hh_b_off' in o:
            W(f"#define L{li}_HH_B_OFF     {o['hh_b_off']}u")
            W(f"#define L{li}_HH_B_NBYTES  {o['hh_b_nbytes']}u")
        else:
            W(f"#define L{li}_HH_B_OFF     0u")
            W(f"#define L{li}_HH_B_NBYTES  0u")
        W(f"#define L{li}_HH_OUT        {hhs[0]}")
        W(f"#define L{li}_HH_IN         {hhs[1]}")
        rq_ih = layer["ih"]["requant"]; rq_hh = layer["hh"]["requant"]
        for g in ("i","f","g","o"):
            W(f"#define L{li}_IH_RQ_M_{g} {rq_ih[g]['mult_q15']}")
            W(f"#define L{li}_IH_RQ_S_{g} {rq_ih[g]['rshift']}")
            W(f"#define L{li}_HH_RQ_M_{g} {rq_hh[g]['mult_q15']}")
            W(f"#define L{li}_HH_RQ_S_{g} {rq_hh[g]['rshift']}")
        W("")
    fc = pkg["fc"]; ofc = pkg["offsets"]["fc"]
    W("// FC head")
    W(f"#define FC_W_OFF {ofc['W_off']}u")
    W(f"#define FC_W_NBYTES {ofc['W_nbytes']}u")
    if "b_off" in ofc:
        W(f"#define FC_B_OFF {ofc['b_off']}u")
        W(f"#define FC_B_NBYTES {ofc['b_nbytes']}u")
    else:
        W(f"#define FC_B_OFF 0u")
        W(f"#define FC_B_NBYTES 0u")
    W(f"#define FC_OUT {fc['W_shape'][0]}")
    W(f"#define FC_IN  {fc['W_shape'][1]}")
    mo_path = OUT_INCLUDE / "model_offsets.h"
    mo_path.write_text("\n".join(mo_lines), encoding="utf-8")
    print(f"[offsets] wrote {mo_path}")

    # --- reference (val/idx as constants) ---
    xb_norm, _ = pick_one_sequence(
        cycles_P, cycles_Tbp, cycles_Tjr,
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        split=REF_SPLIT, seed=SEED, idx_in_split=REF_IDX
    )
    xb_t = torch.from_numpy(xb_norm).unsqueeze(0).to(device)
    if hasattr(model, "quant_eval"):
        cm = model.quant_eval(True)
    else:
        from contextlib import nullcontext; cm = nullcontext()
    with torch.no_grad(), cm:
        yhat_norm = model(xb_t).float().cpu().numpy().item()
    x_q15 = np.vectorize(to_q15, otypes=[np.int16])(xb_norm)
    y_ref_q15 = to_q15(yhat_norm)
    ref_h = OUT_INCLUDE / f"{REF_NAME}.h"
    ref_b = OUT_INCLUDE / f"{REF_NAME}.bin"
    dump_header_ref(ref_h, x_q15, y_ref_q15)
    dump_binary_ref(ref_b, x_q15, y_ref_q15)
    print(f"[done] INT8 export + offsets + reference completed")

if __name__ == "__main__":
    main()
