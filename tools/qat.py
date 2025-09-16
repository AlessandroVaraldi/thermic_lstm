from __future__ import annotations
import json, os, time, math, inspect
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
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------- project-local ----------------
from src.config import (
    # data / model
    CSV_DIR, CSV_GLOB, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    TRAIN_FRAC, VAL_FRAC, WINDOW_SIZE, DT, SEED, MAX_EPOCHS, PATIENCE,
    BATCH_SIZE, LEARNING_RATE, PLOT_DPI, AUG_CYCLES,
    # physics
    LAMBDA_SS, LAMBDA_TR, LAMBDA_WARMUP_EPOCHS,
    RTH_C, RTH_V, C_TH, T_ENV,
    # train defaults
    MIN_EPOCHS_BEST, LR_SCHED_PLATEAU, LR_FACTOR, LR_PATIENCE, LR_MIN,
    EMA_DECAY_QAT, Q_DELAY_UPDATES, Q_FREEZE_UPDATES,
    # runtime 
    DEVICE, AMP_ENABLED, AMP_DTYPE, CKPT, CKPT_CHUNK, TBPTT_K, ACCUM,
    COMPILE, FUSED_ADAM, VAL_INTERVAL, VAL_MAX_BATCHES,
    WORKERS, PIN_MEMORY, PERSIST, PREFETCH,
    MP_TIME, MP_TAU_THR, MP_SCALE_MUL, MP_RSHIFT_DELTA,
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
    """Try TBPTT/ckpt via kwargs, else fallback."""
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
    amp_enabled: bool = AMP_ENABLED, amp_dtype: torch.dtype = torch.bfloat16,
    use_ckpt: bool = CKPT, ckpt_chunk: int = CKPT_CHUNK, tbptt_k: int = TBPTT_K,
    pin: bool = PIN_MEMORY, accum: int = ACCUM, val_interval: int = VAL_INTERVAL, val_max_batches: int | None = None,
    use_fused_adam: bool = FUSED_ADAM
):
    # Optimizer (try fused Adam if supported)
    try:
        opt = torch.optim.Adam(model.parameters(), lr=lr, fused=bool(use_fused_adam))
    except TypeError:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    # LR scheduler (epoch-level)
    sched = ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
        threshold=1e-4, cooldown=1, min_lr=LR_MIN
    ) if LR_SCHED_PLATEAU else None

    # GradScaler (fp16 only)
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))
    devtype = _devtype_from_device(device)

    best_val = float("inf"); best_w = {k:v.detach().clone() for k,v in model.state_dict().items()}
    history = {"train": [], "val": []}
    patience_left = patience
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
            if epoch == 1:
                print(f"[best] best/early-stop abilitati da epoch {max(int(warmup_epochs), MIN_EPOCHS_BEST)}")

            # PI lambda warm-up
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

                # twin-window
                xb_prev = torch.cat([xb[:, :1, :], xb[:, :-1, :]], dim=1)
                xcat    = torch.cat([xb, xb_prev], dim=0)

                with autocast(device_type=devtype, dtype=amp_dtype, enabled=amp_enabled):
                    ycat = forward_model(model, xcat, use_ckpt=use_ckpt, ckpt_chunk=ckpt_chunk, tbptt_k=tbptt_k)
                    B = xb.size(0)
                    y_hat  = ycat[:B]          # (B,) normalized
                    T_prev = ycat[B:]          # (B,) normalized

                    y_hat_deg  = y_hat * std_y_t + mu_y_t       # °C
                    loss_data  = Fnn.mse_loss(y_hat, yb)

                    # last time-step (denorm)
                    x_last = xb[:, -1, :]
                    P_c    = x_last[:, 0] * std_x_t[0] + mu_x_t[0]
                    Tbp_c  = x_last[:, 1] * std_x_t[1] + mu_x_t[1]

                    loss_ss = torch.tensor(0.0, device=device)
                    if lam_ss > 0:
                        Tss = T_steady_state(P_c, Tbp_c, T_ENV, RTH_C, RTH_V)  # °C
                        loss_ss = Fnn.mse_loss(y_hat_deg, Tss)

                    # transient residual
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

                # backward/step
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

            # ---------------- VAL ----------------
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
            gate_epoch = max(int(warmup_epochs), MIN_EPOCHS_BEST)
            ckpt_active = (epoch >= gate_epoch)
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
                tag = "w"
            else:
                tag = "·"

            ep_dt = time.perf_counter() - ep_t0
            if sched is not None and do_val:
                sched.step(val_mse)

            # save ckpts
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
                f"lr={opt.param_groups[0]['lr']:.2e} "
                f"t/ep={fmt_hms(ep_dt)} {tag}",
                flush=True
            )

            if do_val and ckpt_active and patience_left <= 0:
                print(f"Early stopping (best val={best_val:.6f})")
                break

    except KeyboardInterrupt:
        print("User interrupt.")

    model.load_state_dict(best_w)
    # plot curves
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
    seed_everything(SEED)

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
        for _ in range(max(0, AUG_CYCLES)):
            P2, Tbp2, Tjr2 = augment_cycle(P, cols["Tbp"], cols["Tjr"])
            shift = (cycles_t[-1][-1] - cycles_t[-1][0]) + dt
            cycles_t.append(cols["t"] + shift)
            cycles_P.append(P2); cycles_Tbp.append(Tbp2); cycles_Tjr.append(Tjr2)

    n_cycles = len(cycles_P)
    split = split_cycles(n_cycles, TRAIN_FRAC, VAL_FRAC, SEED)

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
    nw  = int(WORKERS)
    pin = bool(PIN_MEMORY)
    per = bool(PERSIST) if nw > 0 else False
    pf  = int(PREFETCH) if nw > 0 else None

    ds_tr = build_concat_dataset(split.train, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)
    ds_va = build_concat_dataset(split.val,   cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)
    ds_te = build_concat_dataset(split.test,  cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, pin_memory=pin, persistent_workers=per, prefetch_factor=pf, drop_last=False)

    # ------------ math kernels (TF32) ------------
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv"):
        if hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.set_float32_matmul_precision("high")

    # ------------ model (QAT INT8) ------------
    device = DEVICE
    model = LSTMModelInt8QAT(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout_p=DROPOUT,
        S_gate_q8=int(getattr(LSTMModelInt8QAT, "DEFAULT_S_GATE_Q8", 32)),   # fallback if needed
        S_tanhc_q8=int(getattr(LSTMModelInt8QAT, "DEFAULT_S_TANHC_Q8", 64)),
    ).to(device)

    # QAT observers
    try:
        from src.qat_int8 import FakeQuant
        for m in model.modules():
            if isinstance(m, FakeQuant):
                m.set_qat_hparams(
                    ema_decay=EMA_DECAY_QAT,
                    quant_delay=Q_DELAY_UPDATES,
                    freeze_after=Q_FREEZE_UPDATES
                )
        print(f"[qat] observers: ema={EMA_DECAY_QAT} q_delay={Q_DELAY_UPDATES} freeze_after={Q_FREEZE_UPDATES}")
    except Exception as _e:
        print(f"[qat] observer tuning skipped: {_e}")

    # temporal mixed precision
    if int(MP_TIME):
        if hasattr(model, "enable_time_mixed_precision"):
            try:
                model.enable_time_mixed_precision(
                    tau_thr=float(MP_TAU_THR),
                    scale_mul=float(MP_SCALE_MUL),
                    rshift_delta=int(MP_RSHIFT_DELTA)
                )
                print(f"[mp-time] enabled: tau_thr={MP_TAU_THR}, scale_mul={MP_SCALE_MUL}, rshift_delta={MP_RSHIFT_DELTA}")
            except Exception as e:
                print(f"[mp-time] skipped: {e}")
        else:
            print("[mp-time] model does not support enable_time_mixed_precision(), skipped")

    # torch.compile (optional)
    if int(COMPILE):
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"[compile] skipped: {e}")

    # ------------ train ------------
    amp_dtype = torch.bfloat16 if str(AMP_DTYPE).lower() == "bf16" else torch.float16
    val_max_batches = None if int(VAL_MAX_BATCHES) <= 0 else int(VAL_MAX_BATCHES)

    model, hist, best_val = train_qat(
        model, dl_tr, dl_va, device,
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        dt=float(np.median([float(np.median(np.diff(cycles_t[i]))) for i in split.train])) if split.train else DT,
        lambda_ss=LAMBDA_SS, lambda_tr=LAMBDA_TR,
        lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, patience=PATIENCE, warmup_epochs=LAMBDA_WARMUP_EPOCHS,
        amp_enabled=bool(AMP_ENABLED), amp_dtype=amp_dtype,
        use_ckpt=bool(CKPT), ckpt_chunk=int(CKPT_CHUNK), tbptt_k=int(TBPTT_K),
        pin=bool(PIN_MEMORY), accum=int(ACCUM), val_interval=int(VAL_INTERVAL), val_max_batches=val_max_batches,
        use_fused_adam=bool(FUSED_ADAM)
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
            amp_enabled=bool(AMP_ENABLED), amp_dtype=amp_dtype
        )
    test_mae = mae(y_pred_test, y_true_test)
    test_rmse= rmse(y_pred_test, y_true_test)
    test_r2  = r2  (y_pred_test, y_true_test)

    print("\n=== Test (denormalized, °C) ===")
    print(f"MSE : {test_mse:.6f}")
    print(f"MAE : {test_mae:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"R²  : {test_r2:.4f}")

    # ------------ ODE baseline (for comparison) ------------
    T_ode_list, ode_time_tot = [], 0.0
    for i in split.test:
        T_ode_i, ode_time = solve_reference_ode(cycles_t[i], cycles_P[i], cycles_Tbp[i])
        T_ode_list.append(T_ode_i); ode_time_tot += ode_time

    # ------------ meta ------------
    meta = {
        "qat": True,
        "hidden": HIDDEN_SIZE, "layers": NUM_LAYERS, "dropout": DROPOUT,
        "S_gate_q8": int(getattr(LSTMModelInt8QAT, "DEFAULT_S_GATE_Q8", 32)),
        "S_tanhc_q8": int(getattr(LSTMModelInt8QAT, "DEFAULT_S_TANHC_Q8", 64)),
        "epochs": MAX_EPOCHS, "batch": BATCH_SIZE, "lr": LEARNING_RATE,
        "lambda_ss": LAMBDA_SS, "lambda_tr": LAMBDA_TR, "warmup": LAMBDA_WARMUP_EPOCHS,
        "mu_x": mu_x.tolist(), "std_x": std_x.tolist(), "mu_y": mu_y, "std_y": std_y,
        "dt_measured": float(np.median([float(np.median(np.diff(t))) for t in cycles_t])) if cycles_t else float(DT),
        "DT_config": float(DT),
        "splits": {"train": split.train, "val": split.val, "test": split.test},
        "metrics": {"val_best_mse": best_val, "test_mse": test_mse, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2},
        "amp": bool(AMP_ENABLED), "amp_dtype": ("bf16" if amp_dtype == torch.bfloat16 else "fp16"), "ckpt": bool(CKPT),
        "ckpt_chunk": int(CKPT_CHUNK), "tbptt_k": int(TBPTT_K),
        "accum": int(ACCUM), "val_interval": int(VAL_INTERVAL), "val_max_batches": (None if VAL_MAX_BATCHES <= 0 else int(VAL_MAX_BATCHES)),
        "compile": int(COMPILE),
        "mp_time": int(MP_TIME), "mp_tau_thr": float(MP_TAU_THR),
        "mp_scale_mul": float(MP_SCALE_MUL), "mp_rshift_delta": int(MP_RSHIFT_DELTA),
        "fused_adam": int(FUSED_ADAM)
    }
    (CKPT_DIR/"meta_qat.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved: {best_path}, {last_path}, {CKPT_DIR/'meta_qat.json'}")

if __name__ == "__main__":
    main()
