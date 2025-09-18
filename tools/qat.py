# qat.py — Quantization-Aware Training script (INT8) for PI-LSTM
# (Rewritten for clarity/robustness without removing features)

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
    # plotting path alias
    PLOTS_DIR as _PLOTS_DIR
)
from src.data_utils import (
    seed_everything, load_all_csvs, compute_powers,
    augment_cycle, solve_reference_ode
)
from src.dataset import WindowDataset
from src.phys_models import T_steady_state, transient_residual
from src.qat_int8 import LSTMModelInt8QAT

# Ensure output dirs exist (use centralized config PLOTS_DIR)
CKPT_DIR = Path("checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = Path(_PLOTS_DIR);   PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    """Retro-compatibility hook: allows passing optional kwargs to forward() if supported."""
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
    Runs the model forward, trying to use TBPTT/checkpoint via kwargs if supported.
    Backward compatibility: if forward() does not accept these kwargs, fallback to checkpoint wrapper.
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

# ====================== QAT robustness helpers ======================
def _maybe_import_fakequant():
    try:
        from src.qat_int8 import FakeQuant
        return FakeQuant
    except Exception:
        return None

@torch.no_grad()
def _recalibrate_qat(model, dl, device, max_steps: int = 64):
    """
    Passa alcuni batch in avanti con observer attivi per aggiornare min/max/scale,
    poi rifreeza se supportato.
    """
    model.train()
    steps = 0
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        _ = model(xb)
        steps += 1
        if steps >= max_steps:
            break
    # rifreeze (se disponibile)
    FakeQuant = _maybe_import_fakequant()
    if FakeQuant is not None:
        for m in model.modules():
            if isinstance(m, FakeQuant):
                if hasattr(m, "freeze"):
                    m.freeze()
    model.eval()

def _print_clamp_stats(model):
    """
    Stampa % di clamp per modulo FakeQuant se l'API lo espone.
    """
    FakeQuant = _maybe_import_fakequant()
    if FakeQuant is None:
        return
    stats = []
    for name, m in model.named_modules():
        if isinstance(m, FakeQuant):
            pct = None
            if hasattr(m, "clamp_ratio"):
                try:
                    pct = float(m.clamp_ratio()) * 100.0
                except Exception:
                    pct = None
            elif hasattr(m, "get_clamp_stats"):
                try:
                    mn, mx, tot = m.get_clamp_stats()
                    if tot > 0:
                        pct = 100.0 * float(mn + mx) / float(tot)
                except Exception:
                    pct = None
            if pct is not None:
                stats.append((name, pct))
    if stats:
        s = " | ".join(f"{n}:{p:.2f}%" for n,p in stats)
        print(f"[clamp] {s}")

# Toggle opzionali (default: disattivati)
ENABLE_INPUT_SOFT_CLIP = True
ENABLE_SCALE_SCAN      = True

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
    use_fused_adam: bool = FUSED_ADAM,
    # nuovi opzionali per robustezza
    mu_x_soft: np.ndarray | None = None, std_x_soft: np.ndarray | None = None, x_clip_sigma: float | None = None,
    lambda_c_reg: float = 0.0
):
    # Optimizer (try fused Adam if supported)
    try:
        opt = torch.optim.Adam(model.parameters(), lr=lr, fused=bool(use_fused_adam))
        if bool(use_fused_adam):
            print("[opt] fused Adam enabled")
    except TypeError:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        if bool(use_fused_adam):
            print("[opt] fused Adam not available on this PyTorch; falling back")

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
    # soft clip tensors (se abilitati)
    if mu_x_soft is not None and std_x_soft is not None and x_clip_sigma is not None:
        mu_x_t_soft  = torch.tensor(mu_x_soft,  device=device, requires_grad=False)
        std_x_t_soft = torch.tensor(std_x_soft, device=device, requires_grad=False)
    else:
        mu_x_t_soft = std_x_t_soft = None

    accum = max(1, int(accum))
    step_mod = 0

    # Info about gating early-stop/best
    gate_epoch_msg = max(int(warmup_epochs), int(MIN_EPOCHS_BEST))
    print(f"[best] best/early-stop enabled from epoch {gate_epoch_msg} "
          f"(warmup_epochs={int(warmup_epochs)}, MIN_EPOCHS_BEST={int(MIN_EPOCHS_BEST)})")

    try:
        for epoch in range(1, max_epochs+1):
            ep_t0 = time.perf_counter()

            # PI lambda warm-up
            scale = min(1.0, epoch / max(1, warmup_epochs)) if warmup_epochs>0 else 1.0
            lam_ss = lambda_ss * scale
            lam_tr = lambda_tr * scale

            # --- Smart refreeze nelle ultime epoche ---
            REFREEZE_START_EPOCH = max(1, max_epochs - 5)
            FakeQuant = _maybe_import_fakequant()
            if FakeQuant is not None and epoch == REFREEZE_START_EPOCH:
                for m in model.modules():
                    if isinstance(m, FakeQuant):
                        try:
                            if hasattr(m, "set_qat_hparams"):
                                m.set_qat_hparams(
                                    ema_decay=min(0.90, EMA_DECAY_QAT),
                                    quant_delay=0,
                                    freeze_after=10
                                )
                            if hasattr(m, "unfreeze"):
                                m.unfreeze()
                        except Exception:
                            pass

            # ---------------- TRAIN ----------------
            model.train()
            loss_sum, n_batches = 0.0, 0
            opt.zero_grad(set_to_none=True)

            for xb, yb in dl_train:
                xb = xb.to(device, non_blocking=pin)
                yb = yb.to(device, non_blocking=pin)

                # Clip morbido degli input normalizzati (opt-in)
                if mu_x_t_soft is not None and std_x_t_soft is not None and x_clip_sigma is not None:
                    with torch.no_grad():
                        lo = (mu_x_t_soft - x_clip_sigma*std_x_t_soft - mu_x_t)/std_x_t
                        hi = (mu_x_t_soft + x_clip_sigma*std_x_t_soft - mu_x_t)/std_x_t
                        xb.clamp_(lo, hi)

                # twin-window: forward on xb and xb_prev (shifted by 1)
                # to obtain in a single pass both y_hat (t) and T_prev (t-1),
                # used in the transient residual loss.
                xb_prev = torch.cat([xb[:, :1, :], xb[:, :-1, :]], dim=1)
                xcat    = torch.cat([xb, xb_prev], dim=0)

                with autocast(device_type=devtype, dtype=amp_dtype, enabled=amp_enabled):
                    ycat = forward_model(model, xcat, use_ckpt=use_ckpt, ckpt_chunk=ckpt_chunk, tbptt_k=tbptt_k)
                    B = xb.size(0)
                    y_hat  = ycat[:B]          # (B,) normalized
                    T_prev = ycat[B:]          # (B,) normalized

                    y_hat_deg  = y_hat * std_y_t + mu_y_t       # °C
                    loss_data  = Fnn.mse_loss(y_hat, yb)        # (normalized space)

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
                    # Regolarizzazione morbida su c_t (se il forward può restituirla)
                    if lambda_c_reg > 0.0 and (_supports_kwargs(model.forward, {"return_state"}) or _supports_kwargs(model.forward, {"return_c"})):
                        try:
                            ycat2 = model(
                                xcat,
                                use_ckpt=use_ckpt, ckpt_chunk=int(ckpt_chunk), tbptt_k=int(tbptt_k),
                                **({"return_state": True} if _supports_kwargs(model.forward, {"return_state"}) else {"return_c": True})
                            )
                            c_t = None
                            if isinstance(ycat2, tuple):
                                for part in ycat2:
                                    if isinstance(part, (list, tuple)) and len(part) == 2:
                                        c_t = part[1]
                                    elif isinstance(part, dict) and "c" in part:
                                        c_t = part["c"]
                            if c_t is not None:
                                C_MAX = torch.tensor(5.0, device=device)
                                loss_reg_c = torch.mean(torch.relu(torch.abs(c_t) - C_MAX) ** 2)
                                loss = loss + float(lambda_c_reg) * loss_reg_c
                        except Exception:
                            pass
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
                # (1) aggiorna observer con val senza quant, no grad
                _cm_obs = model.quant_eval(False) if hasattr(model, "quant_eval") else nullcontext()
                with _cm_obs, torch.no_grad():
                    for i, (xb, yb) in enumerate(dl_val):
                        if val_max_batches is not None and i >= val_max_batches:
                            break
                        _ = model(xb.to(device))
                # (2) misura in quant mode
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
            gate_epoch = max(int(warmup_epochs), int(MIN_EPOCHS_BEST))
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
                tag = "w"  # within warmup gate
            else:
                tag = "·"

            ep_dt = time.perf_counter() - ep_t0
            if sched is not None and do_val:
                sched.step(val_mse)

            # save ckpts (last every epoch, best on improvement)
            try:
                torch.save({"model": model.state_dict()}, last_path)
            except Exception as e:
                print(f"[ckpt] last save skipped: {e}")
            if improved:
                try:
                    torch.save({"model": best_w}, best_path)
                except Exception as e:
                    print(f"[ckpt] best save skipped: {e}")

            # Note (units): val is MSE in normalized space; test metrics in °C
            print(
                f"[{epoch:03d}] train={train_avg:.6f}  val={val_mse:.6f} (norm)  "
                f"λss={lam_ss:.2e} λtr={lam_tr:.2e}  "
                f"lr={opt.param_groups[0]['lr']:.2e} "
                f"t/ep={fmt_hms(ep_dt)} {tag}",
                flush=True
            )

            if do_val and ckpt_active and patience_left <= 0:
                print(f"Early stopping (best val={best_val:.6f} in normalized space)")
                break
            _print_clamp_stats(model)

    except KeyboardInterrupt:
        print("User interrupt.")

    # Restore best weights
    model.load_state_dict(best_w)

    # plot curves
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        xs = range(1, len(history["train"])+1)
        plt.plot(xs, history["train"], label="Train")
        plt.plot(xs, history["val"],   label="Val (norm)")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title("QAT PI-LSTM – loss curves")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(PLOTS_DIR/"loss_curves_qat.png", dpi=PLOT_DPI); plt.close()
    except Exception as e:
        print(f"[plot] skipped: {type(e).__name__}: {e}")

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
    dt_all = float(np.median(dts))
    if (max(dts) - min(dts)) > 1e-6:
        print(f"[warn] dt not uniform across CSVs — median={dt_all:.9f}, range=[{min(dts):.9f},{max(dts):.9f}]")

    for cols in datasets:
        P = compute_powers(cols["Id"], cols["Iq"])
        cycles_t.append(cols["t"]); cycles_P.append(P); cycles_Tbp.append(cols["Tbp"]); cycles_Tjr.append(cols["Tjr"])
        for _ in range(max(0, AUG_CYCLES)):
            P2, Tbp2, Tjr2 = augment_cycle(P, cols["Tbp"], cols["Tjr"])
            shift = (cycles_t[-1][-1] - cycles_t[-1][0]) + dt_all
            cycles_t.append(cols["t"] + shift)
            cycles_P.append(P2); cycles_Tbp.append(Tbp2); cycles_Tjr.append(Tjr2)

    n_cycles = len(cycles_P)
    split = split_cycles(n_cycles, TRAIN_FRAC, VAL_FRAC, SEED)
    print(f"[data] cycles: total={n_cycles}  split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    # ------------ stats (train only) ------------
    PAD = WINDOW_SIZE - 1
    def cat_with_pad(arrs):
        pad = np.full(PAD, np.nan, dtype=float)
        return np.concatenate([np.concatenate([a, pad]) for a in arrs])[:-PAD]

    P_train   = cat_with_pad([cycles_P[i]   for i in split.train]) if split.train else np.array([], dtype=float)
    Tbp_train = cat_with_pad([cycles_Tbp[i] for i in split.train]) if split.train else np.array([], dtype=float)
    Tjr_train = cat_with_pad([cycles_Tjr[i] for i in split.train]) if split.train else np.array([], dtype=float)

    mu_x  = np.nanmean(np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32) if P_train.size else np.array([0.0, 0.0], dtype=np.float32)
    std_x = (np.nanstd (np.column_stack([P_train, Tbp_train]), axis=0).astype(np.float32) + 1e-6) if P_train.size else np.array([1.0, 1.0], dtype=np.float32)
    mu_y  = float(np.nanmean(Tjr_train).astype(np.float32)) if Tjr_train.size else 0.0
    std_y = float(np.nanstd (Tjr_train).astype(np.float32) + 1e-6) if Tjr_train.size else 1.0

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
        # explicit (avoids getattr on class defaults not defined)
        S_gate_q8=32,
        S_tanhc_q8=64,
    ).to(device)

    # (opzionale) mini grid scale per gate/tanh(c)
    if ENABLE_SCALE_SCAN:
        def _dry_run_score(model_try, dl, device, steps=200):
            model_try.train()
            devtype = _devtype_from_device(device)
            opt = torch.optim.SGD(model_try.parameters(), lr=1e-4)
            it = 0
            for xb, yb in dl:
                xb = xb.to(device); yb = yb.to(device)
                with autocast(device_type=devtype, dtype=torch.bfloat16, enabled=True):
                    y = model_try(xb)
                    loss = Fnn.mse_loss(y, yb)
                loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                it += 1
                if it >= steps: break
            _cm = model_try.quant_eval(True) if hasattr(model_try, "quant_eval") else nullcontext()
            with _cm:
                mse_q, _, _ = evaluate_epoch(model_try, dl_va, device, mu_y, std_y, amp_enabled=True, amp_dtype=torch.bfloat16, max_batches=10)
            return mse_q
        candidates = [(32,64), (32,128), (16,64)]
        best_cfg, best_score = None, float("inf")
        for sg, st in candidates:
            m_try = LSTMModelInt8QAT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, S_gate_q8=sg, S_tanhc_q8=st).to(device)
            try:
                sc = _dry_run_score(m_try, dl_tr, device, steps=200)
                print(f"[scale-scan] S_gate={sg} S_tanhc={st} => val_q MSE={sc:.6f}")
                if sc < best_score:
                    best_score, best_cfg = sc, (sg, st)
            except Exception as e:
                print(f"[scale-scan] skip {sg}/{st}: {e}")
        if best_cfg is not None:
            sg, st = best_cfg
            print(f"[scale-scan] scelgo S_gate={sg}, S_tanhc={st}")
            model = LSTMModelInt8QAT(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, S_gate_q8=sg, S_tanhc_q8=st).to(device)

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

    # temporal mixed precision (tanh(c) grid only)
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
                print(f"[mp-time] skipped: {type(e).__name__}: {e}")
        else:
            print("[mp-time] model does not support enable_time_mixed_precision(), skipped")

    # torch.compile (optional)
    if int(COMPILE):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[compile] torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"[compile] skipped: {type(e).__name__}: {e}")

    # ------------ train ------------
    amp_dtype = torch.bfloat16 if str(AMP_DTYPE).lower() == "bf16" else torch.float16
    val_max_batches = None if int(VAL_MAX_BATCHES) <= 0 else int(VAL_MAX_BATCHES)

    # Note dt: we compute dt only on TRAIN cycles for consistency with TR loss
    dt_train = float(np.median([float(np.median(np.diff(cycles_t[i]))) for i in split.train])) if split.train else DT
    print(f"[dt] dt_all={dt_all:.9f}  dt_train={dt_train:.9f} (used for transient residual)")

    model, hist, best_val = train_qat(
        model, dl_tr, dl_va, device,
        mu_x=mu_x, std_x=std_x, mu_y=mu_y, std_y=std_y,
        dt=dt_train,
        lambda_ss=LAMBDA_SS, lambda_tr=LAMBDA_TR,
        lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, patience=PATIENCE, warmup_epochs=LAMBDA_WARMUP_EPOCHS,
        amp_enabled=bool(AMP_ENABLED), amp_dtype=amp_dtype,
        use_ckpt=bool(CKPT), ckpt_chunk=int(CKPT_CHUNK), tbptt_k=int(TBPTT_K),
        pin=bool(PIN_MEMORY), accum=int(ACCUM), val_interval=int(VAL_INTERVAL), val_max_batches=val_max_batches,
        use_fused_adam=bool(FUSED_ADAM),
        # input soft clip (opt-in)
        mu_x_soft=(mu_x if not ENABLE_INPUT_SOFT_CLIP else mu_x),  # placeholder; vedi blocco sotto per soft stats reali
        std_x_soft=(std_x if not ENABLE_INPUT_SOFT_CLIP else std_x),
        x_clip_sigma=(None if not ENABLE_INPUT_SOFT_CLIP else 5.0),
        # reg su c_t (disattivata di default; metti 1e-4 se supportata)
        lambda_c_reg=0.0
    )

    # ------------ save (best + export) ------------
    best_path = CKPT_DIR/"best_qat.pth"; last_path = CKPT_DIR/"last_qat.pth"
    # Gap-check float vs quant su validation + eventuale micro-recalibrazione
    try:
        _cm_f = model.quant_eval(False) if hasattr(model, "quant_eval") else nullcontext()
        with _cm_f:
            mse_float, _, _ = evaluate_epoch(model, dl_va, device, mu_y, std_y,
                                             amp_enabled=bool(AMP_ENABLED), amp_dtype=amp_dtype,
                                             max_batches=val_max_batches)
        _cm_q = model.quant_eval(True) if hasattr(model, "quant_eval") else nullcontext()
        with _cm_q:
            mse_quant, _, _ = evaluate_epoch(model, dl_va, device, mu_y, std_y,
                                             amp_enabled=bool(AMP_ENABLED), amp_dtype=amp_dtype,
                                             max_batches=val_max_batches)
        gap = (mse_quant - mse_float) / max(mse_float, 1e-12)
        print(f"[gap] val MSE quant vs float: {gap*100:.2f}%")
        if gap > 0.10:
            print("[gap] >10%: eseguo micro-recalibrazione observer prima dell'export")
            _recalibrate_qat(model, dl_tr, device, max_steps=64)
    except Exception as e:
        print(f"[gap] skip: {e}")

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
        print(f"[export quant] skipped: {type(e).__name__}: {e}")

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

    _print_clamp_stats(model)
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
        "S_gate_q8": 32,
        "S_tanhc_q8": 64,
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

    # Final save note: at this point *best* has been rewritten;
    # *last* was updated during training at every epoch.
    print(f"\nSaved now: {best_path}, {CKPT_DIR/'meta_qat.json'}")
    print(f"Note: {last_path} was updated every epoch during training.")

if __name__ == "__main__":
    main()
