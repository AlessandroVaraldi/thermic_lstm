from __future__ import annotations
import argparse, json, os, time, math, inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch.utils.checkpoint as ckpt
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# ---------------- project-local ----------------
from src.config import (
    CSV_DIR, CSV_GLOB, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, DT,
    LAMBDA_SS, LAMBDA_TR, LAMBDA_WARMUP_EPOCHS,
    SEED, MAX_EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE,
    RTH_C, RTH_V, C_TH, T_ENV, PLOT_DPI, AUG_CYCLES
)
from src.data_utils import (
    seed_everything, load_all_csvs, compute_powers,
    augment_cycle, solve_reference_ode
)
from src.dataset import WindowDataset
from src.phys_models import T_steady_state, transient_residual
from src.qat_int8 import LSTMModelInt8QAT

CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path("plots");      PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ====================== utils ======================
@dataclass
class Split:
    train: list[int]; val: list[int]; test: list[int]

def split_cycles(n: int, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SEED) -> Split:
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_tr = int(round(train_frac * n))
    n_va = int(round(val_frac   * n))
    return Split(idx[:n_tr].tolist(), idx[n_tr:n_tr+n_va].tolist(), idx[n_tr+n_va:].tolist())

class EmptyDataset(Dataset):
    def __len__(self): return 0
    def __getitem__(self, idx): raise IndexError

def build_concat_dataset(idxs, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y):
    if not idxs: return EmptyDataset()
    ds_list = []
    for i in idxs:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y = cycles_Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    return ConcatDataset(ds_list)

def mse(a,b): return float(np.mean((a-b)**2))
def mae(a,b): return float(np.mean(np.abs(a-b)))
def rmse(a,b): return math.sqrt(mse(a,b))
def r2(a,b):
    ss_res = np.sum((a-b)**2)
    ss_tot = np.sum((b-np.mean(b))**2) + 1e-12
    return float(1.0 - ss_res/ss_tot)

def fmt_hms(seconds: float) -> str:
    s = int(seconds + 0.5); m, s = divmod(s, 60); h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _supports_kwargs(fn, keys):
    try:
        sig = inspect.signature(fn)
        has_var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        return all((k in sig.parameters) or has_var_kw for k in keys)
    except Exception:
        return False

def _devtype_from_device(device: str | torch.device) -> str:
    s = str(device)
    return "cuda" if ("cuda" in s and torch.cuda.is_available()) else "cpu"

def forward_model(model, xb, *, use_ckpt: bool, ckpt_chunk: int, tbptt_k: int):
    """
    Prova a usare TBPTT/ckpt fine-grained se il modello li espone via kwargs.
    Fallback: forward diretto, oppure checkpoint "coarse" dell'intero forward.
    """
    if _supports_kwargs(model.forward, {"use_ckpt", "ckpt_chunk", "tbptt_k"}):
        return model(xb, use_ckpt=use_ckpt, ckpt_chunk=int(ckpt_chunk), tbptt_k=int(tbptt_k))
    if use_ckpt:
        def _fwd(inp): return model(inp)
        return ckpt.checkpoint(_fwd, xb, use_reentrant=False)
    return model(xb)

@torch.no_grad()
def evaluate_epoch(model, dl, device, mu_y, std_y, *, amp_enabled: bool, amp_dtype: torch.dtype, max_batches: int | None = None):
    model.eval()
    devtype = _devtype_from_device(device)
    tot, n = 0.0, 0
    preds, gts = [], []
    with autocast(device_type=devtype, dtype=amp_dtype, enabled=amp_enabled):
        for i, (xb, yb) in enumerate(dl):
            if max_batches is not None and i >= max_batches:
                break
            xb = xb.to(device); yb = yb.to(device)
            y_hat = model(xb)
            tot += Fnn.mse_loss(y_hat, yb, reduction="sum").item()
            n   += yb.numel()
            preds.append((y_hat.float().cpu().numpy() * std_y + mu_y))
            gts.append  ((yb.float().cpu().numpy()     * std_y + mu_y))
    yh = np.concatenate(preds) if preds else np.empty((0,), dtype=float)
    yt = np.concatenate(gts)   if gts   else np.empty((0,), dtype=float)
    return tot/max(n,1), yh, yt

# ====================== training (QAT) ======================
def train_qat(
    model, dl_train, dl_val, device,
    mu_x, std_x, mu_y, std_y,
    *,  # only keyword args below
    dt: float,
    lambda_ss=LAMBDA_SS, lambda_tr=LAMBDA_TR,
    lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    warmup_epochs=LAMBDA_WARMUP_EPOCHS,
    amp_enabled: bool = True, amp_dtype: torch.dtype = torch.bfloat16,
    use_ckpt: bool = False, ckpt_chunk: int = 16, tbptt_k: int = 0,
    pin: bool = True, accum: int = 1, val_interval: int = 1, val_max_batches: int | None = None,
    use_fused_adam: bool = True
):
    # Optimizer
    try:
        opt = torch.optim.Adam(model.parameters(), lr=lr, fused=bool(use_fused_adam))
    except TypeError:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # GradScaler
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    devtype = _devtype_from_device(device)

    best_val = float("inf"); best_w = {k:v.detach().clone() for k,v in model.state_dict().items()}
    history = {"train": [], "val": []}
    patience_left = patience
    # checkpoint paths
    best_path = CKPT_DIR/"best_qat.pth"; last_path = CKPT_DIR/"last_qat.pth"

    mu_x_t = torch.tensor(mu_x,  device=device, requires_grad=False)
    std_x_t= torch.tensor(std_x, device=device, requires_grad=False)
    mu_y_t = torch.tensor(mu_y,  device=device, requires_grad=False)
    std_y_t= torch.tensor(std_y, device=device, requires_grad=False)

    accum = max(1, int(accum))
    step_mod = 0

    try:
        for epoch in range(1, max_epochs+1):
            ep_t0 = time.perf_counter()

            # ----- PI lambda warm-up -----
            scale = min(1.0, epoch / max(1, warmup_epochs)) if warmup_epochs>0 else 1.0
            lam_ss = lambda_ss * scale
            lam_tr = lambda_tr * scale

            # ---------------- TRAIN ----------------
            model.train()
            loss_sum, n_batches = 0.0, 0
            opt.zero_grad(set_to_none=True)

            for xb, yb in dl_train:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)

                # Single-pass twin-window
                xb_prev = torch.cat([xb[:, :1, :], xb[:, :-1, :]], dim=1)
                xcat    = torch.cat([xb, xb_prev], dim=0)

                with autocast(device_type=devtype, dtype=amp_dtype, enabled=amp_enabled):
                    ycat = forward_model(model, xcat, use_ckpt=use_ckpt, ckpt_chunk=ckpt_chunk, tbptt_k=tbptt_k)
                    B = xb.size(0)
                    y_hat  = ycat[:B]          # (B,) normalized
                    T_prev = ycat[B:]          # (B,) normalized

                    y_hat_deg  = y_hat * std_y_t + mu_y_t       # °C
                    loss_data  = Fnn.mse_loss(y_hat, yb)

                    # ---- de-normalize last time-step of current window
                    x_last = xb[:, -1, :]                      # (B,2) normalized
                    P_c    = x_last[:, 0] * std_x_t[0] + mu_x_t[0]
                    Tbp_c  = x_last[:, 1] * std_x_t[1] + mu_x_t[1]

                    loss_ss = torch.tensor(0.0, device=device)
                    if lam_ss > 0:
                        Tss = T_steady_state(P_c, Tbp_c, T_ENV, RTH_C, RTH_V)  # °C
                        loss_ss = Fnn.mse_loss(y_hat_deg, Tss)

                    # ---- transient residual
                    if WINDOW_SIZE >= 2 and lam_tr > 0:
                        T_prev_deg = T_prev * std_y_t + mu_y_t

                        x_prev = xb[:, -2, :]
                        P_p    = x_prev[:, 0] * std_x_t[0] + mu_x_t[0]
                        Tbp_p  = x_prev[:, 1] * std_x_t[1] + mu_x_t[1]

                        r_tr = transient_residual(
                            T_prev_deg, y_hat_deg, P_p, P_c, Tbp_p, Tbp_c, dt, RTH_C, RTH_V, C_TH, T_ENV
                        )
                        loss_tr = torch.mean(r_tr**2)
                    else:
                        loss_tr = torch.tensor(0.0, device=device)

                    loss = loss_data + lam_ss * loss_ss + lam_tr * loss_tr
                    if accum > 1:
                        loss = loss / accum

                # backward/step + optimisation
                scaler.scale(loss).backward()
                step_mod += 1
                if (step_mod % accum) == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                loss_sum += float(loss.detach().cpu()) * (accum if accum > 1 else 1.0)
                n_batches += 1

            train_avg = loss_sum / max(1, n_batches)

            # ---------------- VAL  ----------------
            do_val = (epoch % max(1, val_interval)) == 0
            if do_val:
                _cm = model.quant_eval(True) if hasattr(model, "quant_eval") else nullcontext()
                with _cm:
                    val_mse, _, _ = evaluate_epoch(
                        model, dl_val, device, mu_y, std_y,
                        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                        max_batches=val_max_batches
                    )
            else:
                val_mse = history["val"][-1] if history["val"] else float("inf")

            history["train"].append(train_avg)
            history["val"].append(val_mse)

            # ---------------- CKPT / EARLY STOP + LOG -----------
            ckpt_active = (epoch >= int(warmup_epochs))
            improved = do_val and ckpt_active and (val_mse < best_val - 1e-12)
            if improved:
                best_val = val_mse
                best_w = {k:v.detach().clone() for k,v in model.state_dict().items()}
                patience_left = patience
                tag = "✓"
            elif do_val and ckpt_active:
                patience_left -= 1
                tag = " "
            elif do_val and not ckpt_active:
                tag = "w"  # warmup
            else:
                tag = "·"

            ep_dt = time.perf_counter() - ep_t0

            # ----- SAVE CKPTS -----
            try:
                torch.save({"model": model.state_dict()}, last_path)
            except Exception as e:
                print(f"[ckpt] last save skipped: {e}")
            if improved:
                try:
                    torch.save({"model": best_w}, best_path)
                except Exception as e:
                    print(f"[ckpt] best save skipped: {e}")
            print(
                f"[{epoch:03d}] train={train_avg:.6f}  val={val_mse:.6f}  "
                f"λss={lam_ss:.2e} λtr={lam_tr:.2e}  "
                f"amp={int(amp_enabled)} ckpt={int(use_ckpt)} tbptt={tbptt_k} accum={accum}  "
                f"t/ep={fmt_hms(ep_dt)} {tag}",
                flush=True
            )

            if do_val and ckpt_active and patience_left <= 0:
                print(f"Early stopping (best val={best_val:.6f})")
                break

    except KeyboardInterrupt:
        print("User interrupt.")

    model.load_state_dict(best_w)
    # plot curve
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        xs = range(1, len(history["train"])+1)
        plt.plot(xs, history["train"], label="Train")
        plt.plot(xs, history["val"],   label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("QAT PI-LSTM – loss curves")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(PLOTS_DIR/"loss_curves_qat.png", dpi=PLOT_DPI); plt.close()
    except Exception:
        pass

    return model, history, best_val

# ====================== main ======================
def main():
    ap = argparse.ArgumentParser("QAT training – INT8 attention PI-LSTM (optimized)")
    # dataset / split
    ap.add_argument("--augment", type=int, default=AUG_CYCLES)
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    ap.add_argument("--val-frac",   type=float, default=VAL_FRAC)
    # model / qat scales
    ap.add_argument("--hidden", type=int, default=HIDDEN_SIZE)
    ap.add_argument("--layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--S-gate-q8",  type=int, default=32, help="Q8 scale for gate pre-activations")
    ap.add_argument("--S-tanhc-q8", type=int, default=64, help="Q8 scale for tanh(c)")
    # training hparams
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch",  type=int, default=BATCH_SIZE)
    ap.add_argument("--lr",     type=float, default=LEARNING_RATE)
    ap.add_argument("--lambda-ss", type=float, default=LAMBDA_SS)
    ap.add_argument("--lambda-tr", type=float, default=LAMBDA_TR)
    ap.add_argument("--warmup",    type=int, default=LAMBDA_WARMUP_EPOCHS)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed",   type=int, default=SEED)
    # memory/perf
    ap.add_argument("--amp", type=int, default=1, help="enable AMP mixed precision (1/0)")
    ap.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="AMP dtype")
    ap.add_argument("--ckpt", type=int, default=0, help="activation checkpointing coarse (1/0) — not recommended")
    ap.add_argument("--ckpt-chunk", type=int, default=16, help="(used only if model supports fine-grained ckpt)")
    ap.add_argument("--tbptt-k", type=int, default=0, help="truncate BPTT every K steps if supported (0=off)")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--compile", type=int, default=0, help="torch.compile model (0/1)")
    ap.add_argument("--fused-adam", type=int, default=1, help="use Adam fused if available (0/1)")
    # validation
    ap.add_argument("--val-interval", type=int, default=1, help="validate every K epochs")
    ap.add_argument("--val-max-batches", type=int, default=0, help="limit val batches (0=no limit)")
    # data loading
    ap.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 2)//2)))
    ap.add_argument("--pin-memory", type=int, default=1, help="pin memory for DataLoader (0/1)")
    ap.add_argument("--persist", type=int, default=1, help="persist worker processes (0/1)")
    ap.add_argument("--prefetch", type=int, default=4, help="number of batches to prefetch")
    # temporal mixed-precision (lightweight: only scaling/rshift, no activation changes)
    ap.add_argument("--mp-time", type=int, default=0, help="temporal mixed precision (0/1)")
    ap.add_argument("--mp-tau-thr", type=float, default=0.08, help="transient threshold (°C/s)")
    ap.add_argument("--mp-scale-mul", type=float, default=1.5, help="scale factor for S_gate in mp-time")
    ap.add_argument("--mp-rshift-delta", type=int, default=-1, help="delta right-shift for S_gate in mp-time")

    args = ap.parse_args()

    seed_everything(args.seed)

    # ------------ load CSVs & build cycles ------------
    datasets = load_all_csvs()
    if len(datasets) == 0:
        raise FileNotFoundError(f"No CSV found in {CSV_DIR} (glob: {CSV_GLOB})")

    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []
    dts = [float(np.median(np.diff(cols["t"]))) for cols in datasets]
    dt = float(np.median(dts))
    if (max(dts) - min(dts)) > 1e-6:
        print(f"[warn] dt not uniform across CSVs — median={dt:.9f}, range=[{min(dts):.9f},{max(dts):.9f}]")

    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])
        cycles_t.append(cols["t"]); cycles_P.append(P); cycles_Tbp.append(cols["Tbp"]); cycles_Tjr.append(cols["Tjr"])
        for _ in range(max(0, args.augment)):
            P2, Tbp2, Tjr2 = augment_cycle(P, cols["Tbp"], cols["Tjr"])
            shift = (cycles_t[-1][-1] - cycles_t[-1][0]) + dt
            cycles_t.append(cols["t"] + shift)
            cycles_P.append(P2); cycles_Tbp.append(Tbp2); cycles_Tjr.append(Tjr2)

    n_cycles = len(cycles_P)
    split = split_cycles(n_cycles, args.train_frac, args.val_frac, args.seed)

    # ------------ stats (train only) ------------
    PAD = WINDOW_SIZE - 1
    def cat_with_pad(arrs):
        pad = np.full(PAD, np.nan, dtype=float)
        return np.concatenate([np.concatenate([a, pad]) for a in arrs])[:-PAD]

    P_train   = cat_with_pad([cycles_P[i]   for i in split.train])
    Tbp_train = cat_with_pad([cycles_Tbp[i] for i in split.train])
    Tjr_train = cat_with_pad([cycles_Tjr[i] for i in split.train])

    mu_x  = np.nanmean(np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32)
    std_x = np.nanstd (np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32) + 1e-6
    mu_y  = float(np.nanmean(Tjr_train).astype(np.float32))
    std_y = float(np.nanstd (Tjr_train).astype(np.float32) + 1e-6)

    # ------------ datasets & loaders ------------
    ds_tr = build_concat_dataset(split.train, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)
    ds_va = build_concat_dataset(split.val,   cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)
    ds_te = build_concat_dataset(split.test,  cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)

    nw  = int(args.workers)
    pin = bool(args.pin_memory)
    per = bool(args.persist) if nw > 0 else False
    pf  = int(args.prefetch) if nw > 0 else None

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=False)

    # ------------ math kernels (TF32) ------------
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
        if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

    # ------------ model (QAT INT8) ------------
    device = args.device
    model = LSTMModelInt8QAT(
        input_size=INPUT_SIZE,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout_p=args.dropout,
        S_gate_q8=int(args.S_gate_q8),
        S_tanhc_q8=int(args.S_tanhc_q8),
    ).to(device)

    # temporal mixed precision
    if int(args.mp_time):
        if hasattr(model, "enable_time_mixed_precision"):
            try:
                model.enable_time_mixed_precision(
                    tau_thr=float(args.mp_tau_thr),
                    scale_mul=float(args.mp_scale_mul),
                    rshift_delta=int(args.mp_rshift_delta)
                )
                print(f"[mp-time] enabled: tau_thr={args.mp_tau_thr}, scale_mul={args.mp_scale_mul}, rshift_delta={args.mp_rshift_delta}")
            except Exception as e:
                print(f"[mp-time] skipped: {e}")
        else:
            print("[mp-time] model does not support enable_time_mixed_precision(), skipped")

    # torch.compile (optional)
    if int(args.compile):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[compile] skipped: {e}")

    # ------------ train ------------
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    val_max_batches = None if int(args.val_max_batches) <= 0 else int(args.val_max_batches)

    model, hist, best_val = train_qat(
        model, dl_tr, dl_va, device,
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        dt=dt,
        lambda_ss=args.lambda_ss, lambda_tr=args.lambda_tr,
        lr=args.lr, max_epochs=args.epochs, patience=PATIENCE, warmup_epochs=args.warmup,
        amp_enabled=bool(args.amp), amp_dtype=amp_dtype,
        use_ckpt=bool(args.ckpt), ckpt_chunk=int(args.ckpt_chunk), tbptt_k=int(args.tbptt_k),
        pin=pin, accum=int(args.accum), val_interval=int(args.val_interval), val_max_batches=val_max_batches,
        use_fused_adam=bool(args.fused_adam)
    )

    # ------------ save ------------
    best_path = CKPT_DIR/"best_qat.pth"; last_path = CKPT_DIR/"last_qat.pth"
    torch.save({"model": model.state_dict()}, best_path)

    try:
        if hasattr(model, "export_quant_metadata"):
            qmeta = model.export_quant_metadata()
            (CKPT_DIR/"quant_meta.json").write_text(json.dumps(qmeta, indent=2))
            print(f"[export quant] quant_meta.json saved")
        if hasattr(model, "emit_c_header"):
            hdr = model.emit_c_header(qmeta if 'qmeta' in locals() else None)
            (CKPT_DIR/"model_quant.h").write_text(hdr)
            print(f"[export quant] model_quant.h saved")
    except Exception as e:
        print(f"[export quant] skipped: {e}")

    # ------------ test metrics ------------
    _cm = model.quant_eval(True) if hasattr(model, "quant_eval") else nullcontext()
    with _cm:
        test_mse, y_pred_test, y_true_test = evaluate_epoch(
            model, dl_te, device, mu_y, std_y,
            amp_enabled=bool(args.amp), amp_dtype=amp_dtype
        )
    test_mae = mae(y_pred_test, y_true_test)
    test_rmse= rmse(y_pred_test, y_true_test)
    test_r2  = r2  (y_pred_test, y_true_test)

    print("\n=== Test (denormalized, °C) ===")
    print(f"MSE : {test_mse:.6f}")
    print(f"MAE : {test_mae:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"R²  : {test_r2:.4f}")

    # ------------ EXPORT INT8 (for inference) ------------
    try:
        import numpy as _np
        # 0) Safe DataLoader for calibration: no worker, no pin
        from torch.utils.data import DataLoader as _DL
        _cal_ds = ds_va if len(ds_va) > 0 else ds_tr
        _dl_cal = _DL(_cal_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=False)

        # 1) Fast calibration of pre-att scales (per gate, per layer)
        if hasattr(model, "calibrate_preact_scales"):
            pre_scales = model.calibrate_preact_scales(_dl_cal, device, max_batches=8)
        else:
            raise RuntimeError("model.calibrate_preact_scales() missing")

        # 2) Prepare json and binary package
        bin_path = CKPT_DIR/"model_int8.bin"
        json_path = CKPT_DIR/"model_int8.json"
        offsets = {"layers": [], "fc": {}}
        pkg = {
            "format": "INT8-LSTM-v1",
            "layers": [],
            "fc": {},
            "pre_scales": pre_scales,
            "norm": {"mu_x": mu_x.tolist(), "std_x": std_x.tolist(), "mu_y": mu_y, "std_y": std_y},
        }

        # helper: write bytes and manage offset (in BYTE, little-endian)
        off_bytes = 0
        def _write_arr(fh, tensor, dtype):
            nonlocal off_bytes
            arr = tensor.detach().cpu().numpy().astype(dtype, copy=False)
            if arr.dtype == _np.int32 and arr.dtype.byteorder not in ('<','='):  # force LE for i32
                arr = arr.newbyteorder('<')
            data = arr.tobytes(order="C")
            start = off_bytes
            fh.write(data)
            off_bytes += len(data)
            return start, len(data)

        with open(bin_path, "wb") as fh:
            # Per-layer: ih and hh
            for li, cell in enumerate(model.layers):
                layer_entry = {"idx": li, "ih": {}, "hh": {}, "S_gate_q8": int(cell.sigmoid_q8.S_gate_q8),
                            "S_tanhc_q8": int(cell.tanh_q8_c.S_tanhc_q8),
                            "pre_scales": pre_scales[li]}
                offsets["layers"].append({"idx": li})

                # ---- IH ----
                if not hasattr(model, "quantize_linear_int8") or not hasattr(model, "compute_requant"):
                    raise RuntimeError("quantize_linear_int8/compute_requant missing in model")
                ih_W, ih_b, ih_Sx, ih_Sw = model.quantize_linear_int8(cell.ih)
                layer_entry["ih"]["W_shape"] = list(ih_W.shape)  # [4H, in]
                w_off, w_nbytes = _write_arr(fh, ih_W, _np.int8)
                offsets["layers"][-1]["ih_W_off"] = int(w_off)
                offsets["layers"][-1]["ih_W_nbytes"] = int(w_nbytes)
                if cell.ih.bias is not None:
                    b_off, b_nbytes = _write_arr(fh, ih_b, _np.int32)
                    offsets["layers"][-1]["ih_b_off"] = int(b_off)
                    offsets["layers"][-1]["ih_b_nbytes"] = int(b_nbytes)
                layer_entry["ih"]["Sx"] = float(ih_Sx); layer_entry["ih"]["Sw"] = float(ih_Sw)
                rq_ih = {}
                for k in ("i","f","g","o"):
                    m, r = model.compute_requant(ih_Sx, ih_Sw, pre_scales[li][k])
                    rq_ih[k] = {"mult_q15": int(m), "rshift": int(r)}
                layer_entry["ih"]["requant"] = rq_ih

                # ---- HH ----
                hh_W, hh_b, hh_Sx, hh_Sw = model.quantize_linear_int8(cell.hh)
                layer_entry["hh"]["W_shape"] = list(hh_W.shape)  # [4H, H]
                w_off, w_nbytes = _write_arr(fh, hh_W, _np.int8)
                offsets["layers"][-1]["hh_W_off"] = int(w_off)
                offsets["layers"][-1]["hh_W_nbytes"] = int(w_nbytes)
                if cell.hh.bias is not None:
                    b_off, b_nbytes = _write_arr(fh, hh_b, _np.int32)
                    offsets["layers"][-1]["hh_b_off"] = int(b_off)
                    offsets["layers"][-1]["hh_b_nbytes"] = int(b_nbytes)
                layer_entry["hh"]["Sx"] = float(hh_Sx); layer_entry["hh"]["Sw"] = float(hh_Sw)
                rq_hh = {}
                for k in ("i","f","g","o"):
                    m, r = model.compute_requant(hh_Sx, hh_Sw, pre_scales[li][k])
                    rq_hh[k] = {"mult_q15": int(m), "rshift": int(r)}
                layer_entry["hh"]["requant"] = rq_hh

                pkg["layers"].append(layer_entry)

            # FC (head)
            fc_W, fc_b, fc_Sx, fc_Sw = model.quantize_linear_int8(model.fc)
            pkg["fc"]["W_shape"] = list(fc_W.shape)  # [1, 2H]
            w_off, w_nbytes = _write_arr(fh, fc_W, _np.int8)
            offsets["fc"]["W_off"] = int(w_off); offsets["fc"]["W_nbytes"] = int(w_nbytes)
            if model.fc.bias is not None:
                b_off, b_nbytes = _write_arr(fh, fc_b, _np.int32)
                offsets["fc"]["b_off"] = int(b_off); offsets["fc"]["b_nbytes"] = int(b_nbytes)
            pkg["fc"]["Sx"] = float(fc_Sx); pkg["fc"]["Sw"] = float(fc_Sw)

        pkg["offsets"] = offsets
        (json_path).write_text(json.dumps(pkg, indent=2))
        print(f"[export int8] saved: {bin_path}, {json_path}")
    except Exception as e:
        print(f"[export int8] skipped: {e}")

    # ------------ ODE baseline (for comparison) ------------
    T_ode_list, ode_time_tot = [], 0.0
    for i in split.test:
        T_ode_i, ode_time = solve_reference_ode(cycles_t[i], cycles_P[i], cycles_Tbp[i])
        T_ode_list.append(T_ode_i); ode_time_tot += ode_time

    # ------------ meta ------------
    meta = {
        "qat": True,
        "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
        "S_gate_q8": int(args.S_gate_q8), "S_tanhc_q8": int(args.S_tanhc_q8),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "lambda_ss": args.lambda_ss, "lambda_tr": args.lambda_tr, "warmup": args.warmup,
        "mu_x": mu_x.tolist(), "std_x": std_x.tolist(), "mu_y": mu_y, "std_y": std_y,
        "dt_measured": float(dt), "DT_config": float(DT),
        "splits": {"train": split.train, "val": split.val, "test": split.test},
        "metrics": {"val_best_mse": best_val, "test_mse": test_mse, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2},
        "amp": bool(args.amp), "amp_dtype": args.amp_dtype, "ckpt": bool(args.ckpt),
        "ckpt_chunk": int(args.ckpt_chunk), "tbptt_k": int(args.tbptt_k),
        "accum": int(args.accum), "val_interval": int(args.val_interval), "val_max_batches": val_max_batches,
        "compile": int(args.compile),
        "mp_time": int(args.mp_time), "mp_tau_thr": float(args.mp_tau_thr),
        "mp_scale_mul": float(args.mp_scale_mul), "mp_rshift_delta": int(args.mp_rshift_delta),
        "fused_adam": int(args.fused_adam)
    }
    (CKPT_DIR/"meta_qat.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {best_path}, {last_path}, {CKPT_DIR/'meta_qat.json'}")

if __name__ == "__main__":
    main()