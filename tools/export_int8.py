#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-shot export for INT8 LSTM:
  1) model_int8.bin/json
  2) ie/include/model_offsets.h  (all C macros; no JSON needed)
  3) ie/include/ref_auto.{h,bin} (Q15 reference)
Reads defaults from src.config. Tweak constants below if needed.
"""

from __future__ import annotations
import json, re, struct
from pathlib import Path
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

# ---- minimal knobs (no CLI) ----
CKPT_PATH        = Path("checkpoints/best_qat.pth")
META_PATH        = Path("checkpoints/meta_qat.json")
QUANT_META_PATH  = Path("checkpoints/quant_meta.json")
OUT_CHECKPOINTS  = Path("checkpoints")
OUT_INCLUDE      = Path("ie/include")
REF_NAME         = "ref_auto"
CAL_MAX_BATCHES  = 8
REF_SPLIT        = "val"   # "train"|"val"|"test"
REF_IDX          = 0

# ---------------- utils ----------------
def split_indices(n: int, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac   * n))
    return idx[:n_tr].tolist(), idx[n_tr:n_tr+n_va].tolist(), idx[n_tr+n_va:].tolist()

def build_cycles(augment: int):
    ds = load_all_csvs()
    if len(ds) == 0:
        raise FileNotFoundError(f"No CSVs in {CSV_DIR} (glob: {CSV_GLOB})")
    P, Tbp, Tjr = [], [], []
    for cols in ds:
        p = compute_powers(cols["Id"], cols["Iq"])
        P.append(p); Tbp.append(cols["Tbp"]); Tjr.append(cols["Tjr"])
        for _ in range(max(0, augment)):
            p2, t2, j2 = augment_cycle(p, cols["Tbp"], cols["Tjr"])
            P.append(p2); Tbp.append(t2); Tjr.append(j2)
    return P, Tbp, Tjr

def compute_stats_from_train(P, Tbp, Tjr, seed=SEED):
    tr, _, _ = split_indices(len(P), seed=seed)
    PAD = WINDOW_SIZE - 1
    def cat_with_pad(arrs):
        pad = np.full(PAD, np.nan, dtype=float)
        return np.concatenate([np.concatenate([a, pad]) for a in arrs])[:-PAD]
    px = cat_with_pad([P[i] for i in tr])
    tb = cat_with_pad([Tbp[i] for i in tr])
    jr = cat_with_pad([Tjr[i] for i in tr])
    mu_x  = np.nanmean(np.column_stack([px, tb]), axis=0).astype(np.float32)
    std_x = np.nanstd (np.column_stack([px, tb]), axis=0).astype(np.float32) + 1e-6
    mu_y  = float(np.nanmean(jr).astype(np.float32))
    std_y = float(np.nanstd (jr).astype(np.float32) + 1e-6)
    return (mu_x, std_x, mu_y, std_y)

def infer_dims_from_state(state_dict):
    ih_keys = [k for k in state_dict.keys() if re.search(r"^layers\.\d+\.ih\.weight$", k)]
    if not ih_keys:
        raise RuntimeError("Cannot infer dims: ih.weight not found")
    w = state_dict[sorted(ih_keys)[0]]
    out4H, in_features = w.shape
    H = int(out4H // 4)
    input_size = int(in_features)
    layer_idx = [int(re.findall(r"^layers\.(\d+)\.", k)[0]) for k in state_dict.keys() if k.startswith("layers.")]
    num_layers = (max(layer_idx) + 1) if layer_idx else 1
    return input_size, H, num_layers

def to_q15(x: float) -> int:
    xf = float(max(-0.999969, min(0.999969, x)))
    q = int(round(xf * 32768.0))
    if q >  32767: q =  32767
    if q < -32768: q = -32768
    return q

def pick_one_sequence(P, Tbp, Tjr, mu_x, std_x, mu_y, std_y, split="test", seed=SEED, idx_in_split=0):
    tr, va, te = split_indices(len(P), seed=seed)
    pick = {"train": tr, "val": va, "test": te}[split] or list(range(len(P)))
    ds_list = []
    for i in pick:
        X = np.column_stack([P[i], Tbp[i]]); y = Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    from torch.utils.data import ConcatDataset, DataLoader
    ds = ConcatDataset(ds_list); dl = DataLoader(ds, batch_size=1, shuffle=False)
    idx = max(0, min(idx_in_split, len(ds)-1))
    it = iter(dl); xb, yb = None, None
    for k, batch in enumerate(it):
        if k == idx: xb, yb = batch; break
    if xb is None: xb, yb = next(iter(dl))
    return xb[0].numpy(), float(yb[0].numpy())

def write_ref_header(path: Path, x_q15: np.ndarray, y_q15: int):
    W, D = x_q15.shape
    lines = ["#pragma once",
             f"#define REF_W {W}",
             f"#define REF_D {D}",
             f"static const int16_t ref_y_q15 = {int(y_q15)};",
             f"static const int16_t ref_x_q15[{W*D}] = {{"]
    flat = x_q15.reshape(-1).tolist()
    for i in range(0, len(flat), 16):
        chunk = ", ".join(str(int(v)) for v in flat[i:i+16])
        lines.append("  " + chunk + ("," if (i+16)<len(flat) else ""))
    lines.append("};")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ref] wrote {path}")

def write_ref_bin(path: Path, x_q15: np.ndarray, y_q15: int):
    W, D = x_q15.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHh", 0x31464552, W, D, int(y_q15)))  # 'REF1'
        f.write(x_q15.astype(np.int16, copy=False).tobytes(order="C"))
    print(f"[ref] wrote {path}")

# ---------------- main ----------------
def main():
    seed_everything(SEED)
    device = torch.device(DEVICE)
    OUT_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    OUT_INCLUDE.mkdir(parents=True, exist_ok=True)

    # load checkpoint
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # meta selection
    meta, use = {}, "infer"
    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text()); use = "meta_qat"; print(f"[meta] using {META_PATH}")
    elif QUANT_META_PATH.exists():
        meta = json.loads(QUANT_META_PATH.read_text()); use = "quant_meta"; print(f"[meta] using {QUANT_META_PATH}")
    else:
        print("[meta] none found: inferring from checkpoint + CSV stats")

    # params + stats
    if use == "meta_qat":
        hidden   = int(meta.get("hidden", HIDDEN_SIZE))
        layers   = int(meta.get("layers", NUM_LAYERS))
        dropout  = float(meta.get("dropout", 0.0))
        S_gate_q8  = int(meta.get("S_gate_q8", 32))
        S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
        mu_x = np.array(meta["mu_x"], dtype=np.float32); std_x = np.array(meta["std_x"], dtype=np.float32)
        mu_y = float(meta["mu_y"]); std_y = float(meta["std_y"])
        input_size = INPUT_SIZE
    elif use == "quant_meta":
        hidden   = int(meta.get("hidden", HIDDEN_SIZE))
        layers   = int(meta.get("layers", NUM_LAYERS))
        dropout  = float(meta.get("dropout", 0.0))
        S_gate_q8  = int(meta.get("S_gate_q8", 32))
        S_tanhc_q8 = int(meta.get("S_tanhc_q8", 64))
        if "norm" in meta:
            mu_x = np.array(meta["norm"]["mu_x"], dtype=np.float32); std_x = np.array(meta["norm"]["std_x"], dtype=np.float32)
            mu_y = float(meta["norm"]["mu_y"]); std_y = float(meta["norm"]["std_y"])
        else:
            P, Tbp, Tjr = build_cycles(AUG_CYCLES)
            mu_x, std_x, mu_y, std_y = compute_stats_from_train(P, Tbp, Tjr)
        input_size = int(meta.get("input_size", INPUT_SIZE))
    else:
        input_size, hidden, layers = infer_dims_from_state(state)
        dropout = 0.0
        S_gate_q8, S_tanhc_q8 = 32, 64
        P, Tbp, Tjr = build_cycles(AUG_CYCLES)
        mu_x, std_x, mu_y, std_y = compute_stats_from_train(P, Tbp, Tjr)

    # build model
    model = LSTMModelInt8QAT(
        input_size=input_size, hidden_size=hidden, num_layers=layers,
        dropout_p=dropout, S_gate_q8=S_gate_q8, S_tanhc_q8=S_tanhc_q8
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # calib dataset
    if 'P' not in locals():
        P, Tbp, Tjr = build_cycles(AUG_CYCLES)
    tr, va, _ = split_indices(len(P))
    def _concat(idxs):
        from torch.utils.data import ConcatDataset
        ds_list = []
        for i in idxs:
            X = np.column_stack([P[i], Tbp[i]]); y = Tjr[i]
            ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
        return ConcatDataset(ds_list) if ds_list else None
    ds_va = _concat(va); ds_tr = _concat(tr)
    cal_ds = ds_va if (ds_va is not None and len(ds_va) > 0) else ds_tr
    from torch.utils.data import DataLoader
    cal_loader = DataLoader(cal_ds, batch_size=max(1, BATCH_SIZE), shuffle=False, num_workers=0, pin_memory=False)

    # pre-act calibration
    if not hasattr(model, "calibrate_preact_scales"):
        raise RuntimeError("model.calibrate_preact_scales() missing")
    pre_scales = model.calibrate_preact_scales(cal_loader, device, max_batches=CAL_MAX_BATCHES)
    for li in range(len(pre_scales)):
        for k in ("i","f","g","o"):
            v = float(pre_scales[li][k])
            pre_scales[li][k] = (1e-8 if (not np.isfinite(v) or v <= 0.0) else v)

    # quantize + write bin/json
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
            arr = arr.newbyteorder('<')
        data = arr.tobytes(order="C")
        start = off_bytes
        fh.write(data); off_bytes += len(data)
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
        pkg["fc"]["W_shape"] = list(fc_W.shape)
        w_off, w_nbytes = _write_arr(fh, fc_W, _np.int8)
        offsets["fc"]["W_off"] = int(w_off); offsets["fc"]["W_nbytes"] = int(w_nbytes)
        if model.fc.bias is not None:
            b_off, b_nbytes = _write_arr(fh, fc_b, _np.int32)
            offsets["fc"]["b_off"] = int(b_off); offsets["fc"]["b_nbytes"] = int(b_nbytes)
        pkg["fc"]["Sx"] = float(fc_Sx); pkg["fc"]["Sw"] = float(fc_Sw)

        # FC requant -> Q15
        m_fc, r_fc = model.compute_requant(fc_Sx, fc_Sw, 1.0 / 32768.0)
        pkg["fc"]["requant"] = {"mult_q15": int(m_fc), "rshift": int(r_fc)}

    # atomic writes
    bin_tmp = bin_path.with_suffix(".bin.tmp"); bin_tmp.replace(bin_path)
    pkg["offsets"] = offsets
    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(json.dumps(pkg, indent=2)); json_tmp.replace(json_path)
    print(f"[export int8] saved: {bin_path}, {json_path}")

    # ---- build C header (self-sufficient) ----
    def q15(x: float) -> int: return to_q15(x)

    inv_std0 = 1.0/float(pkg["norm"]["std_x"][0]); inv_std1 = 1.0/float(pkg["norm"]["std_x"][1])
    bx0 = -float(pkg["norm"]["mu_x"][0]) * inv_std0
    bx1 = -float(pkg["norm"]["mu_x"][1]) * inv_std1
    inv_std_y = 1.0/float(pkg["norm"]["std_y"])
    by = -float(pkg["norm"]["mu_y"]) * inv_std_y

    mo_lines = []
    W = mo_lines.append
    L = len(pkg["layers"])
    W("// Auto-generated: model_offsets.h (no JSON required)")
    W("#pragma once")
    W("#include <stdint.h>")
    W("")
    W(f"#define MODEL_LAYERS  {L}")
    W(f"#define MODEL_HIDDEN  ( (int){ pkg['layers'][0]['ih']['W_shape'][0] // 4 } )")
    W(f"#define MODEL_INPUT   {pkg['input_size']}")
    W(f"#define MODEL_WIN     {WINDOW_SIZE}")
    W("")
    # Normalization Q15 (x_norm = a*x + b; y_norm = a*y + b)
    W(f"#define NORM_AX0_Q15 {q15(inv_std0)}")
    W(f"#define NORM_BX0_Q15 {q15(bx0)}")
    W(f"#define NORM_AX1_Q15 {q15(inv_std1)}")
    W(f"#define NORM_BX1_Q15 {q15(bx1)}")
    W(f"#define NORM_AY_Q15  {q15(inv_std_y)}")
    W(f"#define NORM_BY_Q15  {q15(by)}")
    W("")
    # Per-layer blocks
    for li, layer in enumerate(pkg["layers"]):
        o = pkg["offsets"]["layers"][li]
        ihs = layer["ih"]["W_shape"]; hhs = layer["hh"]["W_shape"]
        W(f"// Layer {li}")
        # scales for activations
        W(f"#define L{li}_S_GATE_Q8   {layer['S_gate_q8']}")
        W(f"#define L{li}_S_TANHC_Q8  {layer['S_tanhc_q8']}")
        # IH offsets/shapes
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
        # HH offsets/shapes
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
        # Requants IH/HH (per gate) to Q0.8 pre-activations
        rq_ih = layer["ih"]["requant"]; rq_hh = layer["hh"]["requant"]
        for g in ("i","f","g","o"):
            W(f"#define L{li}_IH_RQ_M_{g} {rq_ih[g]['mult_q15']}")
            W(f"#define L{li}_IH_RQ_S_{g} {rq_ih[g]['rshift']}")
            W(f"#define L{li}_HH_RQ_M_{g} {rq_hh[g]['mult_q15']}")
            W(f"#define L{li}_HH_RQ_S_{g} {rq_hh[g]['rshift']}")
        W("")
    # FC head + requant to Q15
    fc = pkg["fc"]; ofc = pkg["offsets"]["fc"]
    W("// FC head")
    W(f"#define FC_W_OFF    {ofc['W_off']}u")
    W(f"#define FC_W_NBYTES {ofc['W_nbytes']}u")
    if "b_off" in ofc:
        W(f"#define FC_B_OFF    {ofc['b_off']}u")
        W(f"#define FC_B_NBYTES {ofc['b_nbytes']}u")
    else:
        W(f"#define FC_B_OFF    0u")
        W(f"#define FC_B_NBYTES 0u")
    W(f"#define FC_OUT {fc['W_shape'][0]}")
    W(f"#define FC_IN  {fc['W_shape'][1]}")
    W(f"#define FC_RQ_M {fc['requant']['mult_q15']}")
    W(f"#define FC_RQ_S {fc['requant']['rshift']}")
    mo_path = OUT_INCLUDE / "model_offsets.h"
    mo_path.write_text("\n".join(mo_lines), encoding="utf-8")
    print(f"[offsets] wrote {mo_path}")

    # ---- Q15 reference (normalized prediction) ----
    xb_norm, _ = pick_one_sequence(P, Tbp, Tjr, mu_x, std_x, mu_y, std_y,
                                   split=REF_SPLIT, seed=SEED, idx_in_split=REF_IDX)
    xb_t = torch.from_numpy(xb_norm).unsqueeze(0).to(device)
    if hasattr(model, "quant_eval"):
        cm = model.quant_eval(True)
    else:
        from contextlib import nullcontext; cm = nullcontext()
    with torch.no_grad(), cm:
        yhat_norm = model(xb_t).float().cpu().numpy().item()
    x_q15 = np.vectorize(to_q15, otypes=[np.int16])(xb_norm)
    y_q15 = to_q15(yhat_norm)
    write_ref_header(OUT_INCLUDE / f"{REF_NAME}.h", x_q15, y_q15)
    write_ref_bin   (OUT_INCLUDE / f"{REF_NAME}.bin", x_q15, y_q15)

    print("[done] Exported bin/json + C header + reference")

if __name__ == "__main__":
    main()
