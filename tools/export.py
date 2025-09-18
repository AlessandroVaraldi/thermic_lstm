# export.py
# Integer-only exporter for MCU: bins, manifest, model_int8.h and weights.h (arrays).
# Use `#define WEIGHTS_DEFINE` before including weights.h to emit definitions.

from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

# --- project-local imports
from src.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, CSV_DIR, CSV_GLOB, WINDOW_SIZE
from src.qat_int8 import LSTMModelInt8QAT  # model + quant helpers
from src.data_utils import load_all_csvs, compute_powers
from src.dataset import WindowDataset

# ---------------------------- utils ----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def _i8(x):  return np.asarray(x,  dtype=np.int8)
def _i16(x): return np.asarray(x, dtype=np.int16)
def _i32(x): return np.asarray(x, dtype=np.int32)
def _u8(x):  return np.asarray(x,  dtype=np.uint8)

def _to_q(x: float, frac_bits: int, bits: int = 16, signed: bool = True) -> int:
    s = float(1 << frac_bits); v = int(round(float(x) * s))
    if signed:
        lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lo, hi = 0, (1 << bits) - 1
    return int(min(hi, max(lo, v)))

def _gate_indices(H: int) -> Dict[str, slice]:
    return {"i": slice(0*H, 1*H), "f": slice(1*H, 2*H), "g": slice(2*H, 3*H), "o": slice(3*H, 4*H)}

def _align_two_requants(mr_a: Tuple[int,int], mr_b: Tuple[int,int], m_max: int = 32767) -> Tuple[int,int,int,int]:
    m_a, r_a = int(mr_a[0]), int(mr_a[1]); m_b, r_b = int(mr_b[0]), int(mr_b[1])
    r = max(r_a, r_b)
    m_a2, m_b2 = m_a << (r - r_a), m_b << (r - r_b)
    while m_a2 > m_max or m_b2 > m_max:
        r += 1; m_a2 = (m_a2 + 1) >> 1; m_b2 = (m_b2 + 1) >> 1
    return int(m_a2), int(m_b2), int(r), int(m_max)

def _make_sigma_lut_u8(S_pre: float) -> np.ndarray:
    xs = np.arange(-128, 128, dtype=np.int16)
    xfp = xs.astype(np.float64) * float(S_pre)
    y  = 1.0 / (1.0 + np.exp(-xfp))
    q  = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
    return q

def _make_tanh_lut_s8(S_pre: float) -> np.ndarray:
    xs = np.arange(-128, 128, dtype=np.int16)
    xfp = xs.astype(np.float64) * float(S_pre)
    y  = np.tanh(xfp)
    q  = np.clip(np.round(y * 127.0), -128, 127).astype(np.int8)
    return q

def _make_tanh_c_lut_s8(c_frac_bits: int = 12, c_range_abs: float = 8.0) -> np.ndarray:
    idx = np.arange(-32768, 32768, dtype=np.int32)
    c_fp = (idx.astype(np.float64) / float(1 << c_frac_bits))
    c_fp = np.clip(c_fp, -c_range_abs, c_range_abs)
    y  = np.tanh(c_fp)
    q  = np.clip(np.round(y * 127.0), -128, 127).astype(np.int8)
    return q

def _c_array(name: str, ctype: str, arr: np.ndarray, wrap: int = 16) -> str:
    # Emit C initializer with line wrapping
    flat = arr.reshape(-1)
    toks = [str(int(x)) for x in flat]
    lines, line = [], []
    for i, t in enumerate(toks, 1):
        line.append(t)
        if (i % wrap) == 0:
            lines.append(", ".join(line)); line = []
    if line: lines.append(", ".join(line))
    body = ",\n  ".join(lines) if lines else ""
    return f"{ctype} {name}[{len(flat)}] = {{\n  {body}\n}};"

def _c_extern(name: str, ctype: str, arr: np.ndarray) -> str:
    return f"extern {ctype} {name}[{arr.size}];"

# ---------------------------- exporter ----------------------------
def export_int8(
    ckpt_path: Path,
    meta_path: Path | None,
    out_dir: Path,
    device: str = "cpu",
    calib_batches: int = 8,
    c_frac_bits: int = 12,
) -> None:
    out_dir = _ensure_dir(out_dir)

    # 1) Build model and load weights
    model = LSTMModelInt8QAT(
        input_size=int(INPUT_SIZE),
        hidden_size=int(HIDDEN_SIZE),
        num_layers=int(NUM_LAYERS),
        dropout_p=float(DROPOUT),
    ).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval()

    # 2) Load meta (or fallback)
    if meta_path and meta_path.exists():
        meta = json.loads(Path(meta_path).read_text())
        mu_x  = np.asarray(meta["mu_x"], dtype=np.float32)
        std_x = np.asarray(meta["std_x"], dtype=np.float32)
        mu_y  = float(meta["mu_y"])
        std_y = float(meta["std_y"])
        mp    = {"enabled": int(meta.get("mp_time", 0)), "tau_thr": float(meta.get("mp_tau_thr", 0.08))}
    else:
        datasets = load_all_csvs()
        if len(datasets) == 0:
            raise FileNotFoundError(f"No CSV in {CSV_DIR} (glob={CSV_GLOB}).")
        P_all, Tbp_all, Tjr_all = [], [], []
        for cols in datasets:
            P_all.append(compute_powers(cols["Id"], cols["Iq"]))
            Tbp_all.append(cols["Tbp"]); Tjr_all.append(cols["Tjr"])
        P_all   = np.concatenate(P_all); Tbp_all = np.concatenate(Tbp_all); Tjr_all = np.concatenate(Tjr_all)
        mu_x  = np.array([np.mean(P_all), np.mean(Tbp_all)], dtype=np.float32)
        std_x = np.array([np.std(P_all)+1e-6, np.std(Tbp_all)+1e-6], dtype=np.float32)
        mu_y  = float(np.mean(Tjr_all)); std_y = float(np.std(Tjr_all)+1e-6)
        mp    = {"enabled": 0, "tau_thr": 0.08}

    # 3) Tiny calibration dl (first CSV)
    try:
        from torch.utils.data import DataLoader
        cols0 = load_all_csvs()[0]
        Xf = np.column_stack([compute_powers(cols0["Id"], cols0["Iq"]), cols0["Tbp"]]).astype(np.float32)
        yf = cols0["Tjr"].astype(np.float32)
        dl = DataLoader(WindowDataset(Xf, yf, mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y), batch_size=16, shuffle=False)
    except Exception:
        dl = []

    # 4) Calibrate S_pre[i/f/g/o] per layer
    with model.quant_eval(True):
        pre_scales = model.calibrate_preact_scales(dl, device, max_batches=int(calib_batches))

    # 5) Quantize linears + gate requants
    H = int(HIDDEN_SIZE); D = int(INPUT_SIZE); L = int(NUM_LAYERS)
    gate_off = _gate_indices(H)

    layers_pack = []
    for li, cell in enumerate(model.layers):
        Wih_s8, Bih_i32, Sx_ih, Sw_ih = model.quantize_linear_int8(cell.ih)
        Whh_s8, Bhh_i32, Sx_hh, Sw_hh = model.quantize_linear_int8(cell.hh)
        Wih_s8 = Wih_s8.cpu().numpy(); Whh_s8 = Whh_s8.cpu().numpy()
        Bih_i32 = Bih_i32.cpu().numpy(); Bhh_i32 = Bhh_i32.cpu().numpy()
        S_pre = pre_scales[li]

        gate_params = {}
        for g in ("i","f","g","o"):
            sL = gate_off[g]
            m_ih, r_ih = model.compute_requant(Sx_ih, Sw_ih, S_pre[g])
            m_hh, r_hh = model.compute_requant(Sx_hh, Sw_hh, S_pre[g])
            m_ih2, m_hh2, r_c, _ = _align_two_requants((m_ih,r_ih), (m_hh,r_hh))
            b_comb = ((Bih_i32[sL] * m_ih2).astype(np.int64) + (Bhh_i32[sL] * m_hh2).astype(np.int64))
            b_comb = (b_comb >> r_c).astype(np.int32)
            gate_params[g] = {
                "m_ih_q15":  _i16(np.full(H, m_ih2)),
                "m_hh_q15":  _i16(np.full(H, m_hh2)),
                "rshift":    _u8 (np.full(H, r_c)),
                "bias_q":    _i32(b_comb),
                "rows":      (int(sL.start), int(sL.stop)),
            }

        lut_sigma = {g: _make_sigma_lut_u8(S_pre[g]) for g in ("i","f","o")}
        lut_tanhg = _make_tanh_lut_s8(S_pre["g"])

        layers_pack.append({
            "li": li,
            "in_dim": (D if li == 0 else H),
            "Wih_s8": Wih_s8, "Whh_s8": Whh_s8,
            "gate_params": gate_params,
            "lut_sigma_i": lut_sigma["i"], "lut_sigma_f": lut_sigma["f"], "lut_sigma_o": lut_sigma["o"],
            "lut_tanh_g": lut_tanhg,
        })

    # 6) FC head
    Wfc_s8, Bfc_i32, Sx_fc, Sw_fc = model.quantize_linear_int8(model.fc)
    Wfc_s8 = Wfc_s8.cpu().numpy(); Bfc_i32 = Bfc_i32.cpu().numpy()
    S_pre_fc = (1.0 / 127.0)
    m_fc, r_fc = model.compute_requant(Sx_fc, Sw_fc, S_pre_fc)

    # 7) tanh(c) LUT
    lut_tanh_c = _make_tanh_c_lut_s8(c_frac_bits=c_frac_bits, c_range_abs=8.0)

    # 8) Save previous artifacts (npz, manifest, bins, model_int8.h)
    out_bin = _ensure_dir(out_dir / "bin")
    out_hdr = out_dir / "model_int8.h"
    out_npz = out_dir / "lstm_int8_pack.npz"
    manifest_path = out_dir / "manifest.json"
    weights_h_path = out_dir / "weights.h"

    # --- NPZ
    npz_dict = {
        "H": H, "D": D, "L": L, "WINDOW": int(WINDOW_SIZE),
        "mu_x": mu_x, "std_x": std_x, "mu_y": np.float32(mu_y), "std_y": np.float32(std_y),
        "mp_enabled": np.uint8(mp["enabled"]), "mp_tau_thr": np.float32(mp["tau_thr"]),
        "c_frac_bits": np.uint8(c_frac_bits),
        "Wfc_s8": Wfc_s8, "Bfc_i32": Bfc_i32, "m_fc_q15": np.int16(m_fc), "r_fc": np.uint8(r_fc),
        "lut_tanh_c": lut_tanh_c,
    }
    for lp in layers_pack:
        li = lp["li"]
        npz_dict[f"L{li}_Wih_s8"] = lp["Wih_s8"]
        npz_dict[f"L{li}_Whh_s8"] = lp["Whh_s8"]
        for g in ("i","f","g","o"):
            gp = lp["gate_params"][g]
            npz_dict[f"L{li}_{g}_m_ih_q15"] = gp["m_ih_q15"]
            npz_dict[f"L{li}_{g}_m_hh_q15"] = gp["m_hh_q15"]
            npz_dict[f"L{li}_{g}_rshift"]   = gp["rshift"]
            npz_dict[f"L{li}_{g}_bias_q"]   = gp["bias_q"]
        npz_dict[f"L{li}_lut_sigma_i"] = lp["lut_sigma_i"]
        npz_dict[f"L{li}_lut_sigma_f"] = lp["lut_sigma_f"]
        npz_dict[f"L{li}_lut_sigma_o"] = lp["lut_sigma_o"]
        npz_dict[f"L{li}_lut_tanh_g"]  = lp["lut_tanh_g"]
    np.savez_compressed(out_npz, **npz_dict)

    # --- Binaries + manifest (still keep for tooling)
    def dump_bin(name: str, arr: np.ndarray) -> str:
        p = out_bin / name; arr.tofile(p); return name
    bin_index = []
    for lp in layers_pack:
        li = lp["li"]
        bin_index.append(("L%d_Wih_s8" % li, dump_bin(f"L{li}_Wih_s8.bin", _i8(lp["Wih_s8"]))))
        bin_index.append(("L%d_Whh_s8" % li, dump_bin(f"L{li}_Whh_s8.bin", _i8(lp["Whh_s8"]))))
        for g in ("i","f","g","o"):
            gp = lp["gate_params"][g]
            bin_index.append((f"L{li}_{g}_m_ih_q15", dump_bin(f"L{li}_{g}_m_ih_q15.bin", _i16(gp["m_ih_q15"])) ))
            bin_index.append((f"L{li}_{g}_m_hh_q15", dump_bin(f"L{li}_{g}_m_hh_q15.bin", _i16(gp["m_hh_q15"])) ))
            bin_index.append((f"L{li}_{g}_rshift",   dump_bin(f"L{li}_{g}_rshift.bin",   _u8 (gp["rshift"]))   ))
            bin_index.append((f"L{li}_{g}_bias_q",   dump_bin(f"L{li}_{g}_bias_q.bin",   _i32(gp["bias_q"]))   ))
        bin_index.append((f"L{li}_lut_sigma_i", dump_bin(f"L{li}_lut_sigma_i_u8.bin", _u8(lp["lut_sigma_i"])) ))
        bin_index.append((f"L{li}_lut_sigma_f", dump_bin(f"L{li}_lut_sigma_f_u8.bin", _u8(lp["lut_sigma_f"])) ))
        bin_index.append((f"L{li}_lut_sigma_o", dump_bin(f"L{li}_lut_sigma_o_u8.bin", _u8(lp["lut_sigma_o"])) ))
        bin_index.append((f"L{li}_lut_tanh_g",  dump_bin(f"L{li}_lut_tanh_g_s8.bin",  _i8(lp["lut_tanh_g"]))  ))
    bin_index.append(("Wfc_s8", dump_bin("FC_W_s8.bin", _i8(Wfc_s8))))
    bin_index.append(("Bfc_i32", dump_bin("FC_B_i32.bin", _i32(Bfc_i32))))
    bin_index.append(("m_fc_q15", dump_bin("FC_m_q15.bin", _i16(np.array([m_fc], dtype=np.int16)))))
    bin_index.append(("r_fc",     dump_bin("FC_rshift.bin", _u8(np.array([r_fc], dtype=np.uint8)))))
    bin_index.append(("tanh_c",   dump_bin("tanh_c_q_lut_s8.bin", _i8(lut_tanh_c))))

    manifest = {
        "H": H, "D": D, "L": L, "WINDOW": int(WINDOW_SIZE),
        "mu_x": mu_x.tolist(), "std_x": std_x.tolist(), "mu_y": float(mu_y), "std_y": float(std_y),
        "c_frac_bits": int(c_frac_bits),
        "pre_order": ["i","f","g","o"],
        "bins": [{"name": k, "file": v} for (k,v) in bin_index],
        "mp": {"enabled": int(mp["enabled"]), "tau_thr_q12": _to_q(mp["tau_thr"], 12, bits=16, signed=False), "suggest_interp": 1}
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- model_int8.h (shapes + FC requant + MP flags)
    lines = []
    P = lines.append
    P("// Auto-generated — integer-only LSTM export (shapes/const)")
    P("#pragma once")
    P("#include <stdint.h>")
    P("")
    P(f"#define LSTM_LAYERS   {L}")
    P(f"#define MODEL_HIDDEN  {H}")
    P(f"#define MODEL_INPUTS  {D}")
    P(f"#define WIN_SIZE      {int(WINDOW_SIZE)}")
    P("")
    P(f"#define C_FRAC_BITS   {int(c_frac_bits)}")  # c(t) Q(3.C_FRAC_BITS)
    P("")
    tau_q12 = _to_q(mp["tau_thr"], 12, bits=16, signed=False)
    P(f"#define MP_TIME_ENABLED  {1 if mp['enabled'] else 0}")
    P(f"#define MP_TAU_THR_Q12   {tau_q12}  // |Δx_norm| threshold in Q4.12")
    P("")
    P("// FC requant to Q1.7")
    P(f"#define FC_M_Q15   {int(m_fc)}")
    P(f"#define FC_RSHIFT  {int(r_fc)}")
    P("")
    out_hdr.write_text("\n".join(lines))

    # 9) weights.h (arrays C; extern by default, defs if WEIGHTS_DEFINE)
    WH = []
    W = WH.append
    W("// Auto-generated — integer-only LSTM weights")
    W("#pragma once")
    W("#include <stdint.h>")
    W('#include "model_int8.h"  // shapes/const')
    W("")
    W("// Define WEIGHTS_DEFINE in exactly one TU to emit definitions.")
    W("#ifdef WEIGHTS_DEFINE")
    emit_defs = True
    def _emit(name, ctype, arr):
        W(_c_array(name, f"const {ctype}", arr))
    def _decl(name, ctype, arr):
        pass
    else_block = []
    else_block.append("#else")
    emit_defs = False

    # we collect both defs and externs in one pass
    arrays = []
    for lp in layers_pack:
        li = lp["li"]; IN = int(lp["in_dim"])
        arrays.append( (f"L{li}_Wih_s8", "int8_t", _i8(lp["Wih_s8"])) )
        arrays.append( (f"L{li}_Whh_s8", "int8_t", _i8(lp["Whh_s8"])) )
        for g in ("i","f","g","o"):
            gp = lp["gate_params"][g]
            arrays.append( (f"L{li}_{g}_m_ih_q15", "int16_t", _i16(gp["m_ih_q15"])) )
            arrays.append( (f"L{li}_{g}_m_hh_q15", "int16_t", _i16(gp["m_hh_q15"])) )
            arrays.append( (f"L{li}_{g}_rshift",   "uint8_t", _u8 (gp["rshift"]))   )
            arrays.append( (f"L{li}_{g}_bias_q",   "int32_t", _i32(gp["bias_q"]))   )
        arrays.append( (f"L{li}_lut_sigma_i", "uint8_t", _u8(lp["lut_sigma_i"])) )
        arrays.append( (f"L{li}_lut_sigma_f", "uint8_t", _u8(lp["lut_sigma_f"])) )
        arrays.append( (f"L{li}_lut_sigma_o", "uint8_t", _u8(lp["lut_sigma_o"])) )
        arrays.append( (f"L{li}_lut_tanh_g",  "int8_t",  _i8(lp["lut_tanh_g"]))  )
    arrays.append( ("FC_W_s8", "int8_t",  _i8(Wfc_s8)) )
    arrays.append( ("FC_B_i32","int32_t", _i32(Bfc_i32)) )
    arrays.append( ("TANH_C_Q_LUT_S8","int8_t", _i8(lut_tanh_c)) )

    # emit definitions
    W("#ifdef WEIGHTS_DEFINE")
    for name, ctype, arr in arrays:
        W(_c_array(name, f"const {ctype}", arr))
        W("")
    W("#else")
    # emit externs
    for name, ctype, arr in arrays:
        W(_c_extern(name, f"const {ctype}", arr))
    W("#endif")
    W("")

    weights_h_path.write_text("\n".join(WH))

    # 10) Done
    print(f"[ok] Wrote: {out_npz}")
    print(f"[ok] Wrote: {manifest_path}")
    print(f"[ok] Wrote: {out_hdr}")
    print(f"[ok] Wrote: {weights_h_path}")
    print(f"[ok] Wrote binaries under: {out_bin}")

# ---------------------------- CLI ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Export INT8 LSTM for MCU (integer-only) + weights.h arrays.")
    ap.add_argument("--ckpt", type=Path, default=Path("checkpoints/best_qat.pth"))
    ap.add_argument("--meta", type=Path, default=Path("checkpoints/meta_qat.json"))
    ap.add_argument("--out",  type=Path, default=Path("export"))
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--calib-batches", type=int, default=8)
    ap.add_argument("--c-frac-bits", type=int, default=12)  # Q3.12 for c(t)
    args = ap.parse_args()

    export_int8(
        ckpt_path=args.ckpt,
        meta_path=args.meta,
        out_dir=args.out,
        device=args.device,
        calib_batches=int(args.calib_batches),
        c_frac_bits=int(args.c_frac_bits),
    )

if __name__ == "__main__":
    main()
