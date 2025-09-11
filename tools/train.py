# ============================================================================
# 1. Standard library imports
# ============================================================================
import argparse
import logging
import math
import os
import time
from functools import partial
from pathlib import Path

# ============================================================================
# 2. Third-party imports
# ============================================================================
import numpy as np
import matplotlib.pyplot as plt
import requests                           # Telegram API
import torch
from torch.utils.data import ConcatDataset, DataLoader

# Optional: AMP and profiling tools
from torch.amp import GradScaler, autocast

try:
    import optuna
except ImportError as e:
    raise ImportError("Optuna not found – install with `pip install optuna`") from e

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# ============================================================================
# 3. Project-local imports
# ============================================================================
from src.config import *  # global settings like SEED, BATCH_SIZE, etc.
from src.data_utils import (
    load_all_csvs, compute_powers, augment_cycle, seed_everything,
    solve_reference_ode, sliding_windows, predict_mc
)
from src.dataset import WindowDataset
from src.models import LSTMModel
from src.train import train_model

import importlib  # used for runtime config updates


CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ---------------- metriche addizionali ---------------------------------
def mae(y_pred, y_true):
    return float(np.mean(np.abs(y_pred - y_true)))

def huber(y_pred, y_true, delta=1.0):
    err  = y_pred - y_true
    mask = np.abs(err) <= delta
    return float(
        np.mean(0.5 * err[mask]**2) + np.mean(delta * (np.abs(err[~mask]) - 0.5*delta))
    )

def gaussian_nll(mu, sigma, y_true, eps=1e-6):
    sigma = np.clip(sigma, eps, None)
    return float(np.mean(
        0.5 * np.log(2 * math.pi * sigma**2) + 0.5 * ((y_true - mu) / sigma)**2
    ))

# ---------------------------------------------------------------------
# 4. Logging helpers
# ---------------------------------------------------------------------
def setup_logging(verbose: bool, log_dir: str = "logs") -> logging.Logger:
    """
    Configure a root logger with file and (optional) console handlers,
    ensuring we do _not_ duplicate StreamHandlers that other libraries
    (e.g. Optuna) may have already installed.
    """
    Path(log_dir).mkdir(exist_ok=True)
    log_path = Path(log_dir) / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for h in logger.handlers[:]:
        logger.removeHandler(h)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Logger initialised – log file: %s", log_path)
    return logger


# ---------------------------------------------------------------------
# 5. Utility functions
# ---------------------------------------------------------------------
def predict(model, dl, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def concat_with_pad(arr_list, pad_len):
    if len(arr_list) == 0:
        return np.empty(0, dtype=float)
    pad = np.full(pad_len, np.nan, dtype=arr_list[0].dtype)
    return np.concatenate([np.concatenate([a, pad]) for a in arr_list])[:-pad_len]


def build_concat_dataset(
    idxs, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y
):
    ds_list = []
    for i in idxs:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y = cycles_Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    return ConcatDataset(ds_list)


# ---------------------------------------------------------------------
# 6. Telegram helpers (unchanged)
# ---------------------------------------------------------------------
def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "Markdown"
) -> bool:
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logging.warning("Telegram token / chat_id missing – notification skipped")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }

    if parse_mode is not None:
        payload["parse_mode"] = parse_mode

    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
        logging.info("Notification sent to Telegram chat %s", chat_id)
        return True
    except requests.RequestException as e:
        logging.error("Telegram notification failed: %s", e)
        return False


# ---------------------------------------------------------------------
# 7. Main pipeline
# ---------------------------------------------------------------------
def main(
    n_trials: int,
    verbose: bool,
    device: str = "cuda",
    use_amp: bool = True,
    use_telegram: bool = True,
    # -----------------------------------------------------------------
    # new switches for the first (architecture) search
    # -----------------------------------------------------------------
    search_hidden: bool = True,
    search_layers: bool = True,
    search_lr: bool = True,
    search_ss: bool = True,
    search_tr: bool = True,
    fixed_hidden: int = 16,
    fixed_layers: int = 1,
    fixed_lr: float | None = None,
    fixed_ss: float | None = None,
    fixed_tr: float | None = None
):

    if use_telegram:
        send_telegram_message(
            f"Starting LSTM training with {n_trials} trials",
            parse_mode="MarkdownV2"
        )

    logger = setup_logging(verbose)

    # -------------------------------------------------- Reproducibility
    seed_everything(SEED)
    logger.info("Random seed set to %s", SEED)

    # -------------------------------------------------- Data loading – MULTI-CSV
    datasets = load_all_csvs()
    if len(datasets) == 0:
        raise RuntimeError("Nessun CSV trovato – controlla CSV_DIR / CSV_GLOB")

    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []

    dt = datasets[0]["t"][1] - datasets[0]["t"][0]
    
    importlib.import_module("src.config").DT = float(dt)
    logger.info("Found %d CSV files – global dt = %.6f s", len(datasets), dt)

    for data_idx, cols in enumerate(datasets):
        P_orig      = compute_powers(cols["Id"], cols["Iq"])
        t_base      = cols["t"]
        cycle_span  = t_base[-1] - t_base[0]

        # -------- ciclo reale
        cycles_t  .append(t_base.copy())
        cycles_P  .append(P_orig.copy())
        cycles_Tbp.append(cols["Tbp"].copy())
        cycles_Tjr.append(cols["Tjr"].copy())

        # -------- augmentation
        for i in range(AUG_CYCLES):
            P_aug, Tbp_aug, Tjr_aug = augment_cycle(P_orig, cols["Tbp"], cols["Tjr"])
            t_shift = (i + 1) * (cycle_span + dt)
            cycles_t .append(t_base + t_shift)
            cycles_P .append(P_aug)
            cycles_Tbp.append(Tbp_aug)
            cycles_Tjr.append(Tjr_aug)

    logger.info(
        "Generated %d cycles (%d real + %d augmented)",
        len(cycles_t), len(datasets), len(cycles_t) - len(datasets)
    )

    n_cycles = len(cycles_t)
    n_train  = int(TRAIN_FRAC * n_cycles)
    n_val    = int(VAL_FRAC   * n_cycles)

    idx_train = list(range(0,               n_train))
    idx_val   = list(range(n_train,         n_train + n_val))
    idx_test  = list(range(n_train + n_val, n_cycles))

    PAD = WINDOW_SIZE - 1  # pad between cycles when concatenating

    # Flattened signals (needed for μ/σ)
    P_train  = concat_with_pad([cycles_P[i]   for i in idx_train], PAD)
    Tbp_train = concat_with_pad([cycles_Tbp[i] for i in idx_train], PAD)

    X_train_raw = np.column_stack([P_train, Tbp_train])
    mu_x, std_x = np.nanmean(X_train_raw, axis=0), np.nanstd(X_train_raw, axis=0)

    Tjr_train = concat_with_pad([cycles_Tjr[i] for i in idx_train], PAD)
    mu_y, std_y = np.nanmean(Tjr_train), np.nanstd(Tjr_train)
    logger.info("Computed normalisation statistics")

    # Datasets / DataLoaders
    ds_train = build_concat_dataset(
        idx_train, cycles_P, cycles_Tbp, cycles_Tjr,
        mu_x, std_x, mu_y, std_y
    )
    ds_val   = build_concat_dataset(
        idx_val, cycles_P, cycles_Tbp, cycles_Tjr,
        mu_x, std_x, mu_y, std_y
    )
    ds_test  = build_concat_dataset(
        idx_test, cycles_P, cycles_Tbp, cycles_Tjr,
        mu_x, std_x, mu_y, std_y
    )

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, drop_last=True)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE)

    logger.info("Using device: %s", device)

    # -------------------------------------------------- 7.1 Optuna studies
    # ---------------------------------------------------------------------
    # 1) Architecture + (optional) learning-rate search
    # ---------------------------------------------------------------------

    any_param_search = search_hidden or search_layers or search_lr or search_ss or search_tr
    if any_param_search:
        logger.info(
            "Starting architecture + LR tuning – searching: hidden=%s, layers=%s, lr=%s, ss=%s, tr=%s",
            search_hidden, search_layers, search_lr, search_ss, search_tr
        )

        def objective_arch(trial: optuna.Trial):
            # choose or fix each parameter
            hidden_size = trial.suggest_int("hidden_size", 8, 32, step=8) if search_hidden else fixed_hidden
            num_layers  = trial.suggest_int("num_layers", 1, 3)           if search_layers else fixed_layers
            lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True) if search_lr else (fixed_lr or LEARNING_RATE)
            lambda_ss   = 0.0
            lambda_tr   = 0.0

            importlib.import_module("src.config").LEARNING_RATE = lr

            model = LSTMModel(INPUT_SIZE, hidden_size, num_layers).to(device)
            model = train_model(
                model, dl_train, dl_val,
                lambda_ss=lambda_ss, lambda_tr=lambda_tr,
                device=device, use_amp=use_amp, logger=logger
            )

            yh_val, y_val = predict(model, dl_val, device)
            yh_val = mu_y + std_y * yh_val
            y_val  = mu_y + std_y * y_val
            val_mse = float(np.mean((yh_val - y_val) ** 2))

            # --------------- checkpoint --------------------------
            study_name = trial.study.study_name        # "lstm_arch"
            ckpt_path  = CKPT_DIR / f"{study_name}_best.pt"

            # NB: study.best_value è None finché non esiste un best
            if (trial.number == 0) or (val_mse < trial.study.best_value):
                torch.save(model.state_dict(), ckpt_path)
                trial.set_user_attr("ckpt_path", str(ckpt_path))

            return float(np.mean((yh_val - y_val) ** 2))

        study_arch = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name="lstm_std"
        )
        study_arch.optimize(
            objective_arch,
            n_trials=n_trials,
            callbacks=[lambda s, t: send_telegram_message(f"Arch trial {t.number}: {t.value:.3e}", parse_mode=None)]
        )

        best_arch = study_arch.best_params
        # Fill in fixed values for parameters not searched
        if not search_hidden:
            best_arch["hidden_size"] = fixed_hidden
        if not search_layers:
            best_arch["num_layers"] = fixed_layers
        if not search_lr:
            best_arch["lr"] = fixed_lr or LEARNING_RATE
        if not search_ss:
            best_arch["lambda_ss"] = fixed_ss if fixed_ss is not None else 0.0
        if not search_tr:
            best_arch["lambda_tr"] = fixed_tr if fixed_tr is not None else 0.0

    else:
        # No search: just use the fixed values directly
        best_arch = dict(hidden_size=fixed_hidden,
                         num_layers=fixed_layers,
                         lr=fixed_lr or LEARNING_RATE,
                         lambda_ss=fixed_ss if fixed_ss is not None else 0.0,
                         lambda_tr=fixed_tr if fixed_tr is not None else 0.0)
        logger.info("No architecture tuning requested – using %s", best_arch)

    logger.info("Best architecture (pre-PI): %s", best_arch)

    # ---------------------------------------------------------------------
    # 2) λ_ss search (λ_tr = 0)
    # -----------------------------------------
    if best_arch.get("lambda_ss") is not None:
        logger.info("λ_ss is fixed to %.3e – skipping λ_ss search", best_arch["lambda_ss"])
        best_lambda_ss = best_arch["lambda_ss"]

    else:
        logger.info("Starting λ_ss tuning (%d trials)…", n_trials)

        import gc
        def objective_lambda_ss(trial: optuna.Trial):
            try:
                lambda_ss = trial.suggest_float("lambda_ss", 1e-7, 1e-5, log=True)
                lambda_tr = 0.0

                model = LSTMModel(
                    INPUT_SIZE,
                    best_arch["hidden_size"],
                    best_arch["num_layers"]
                ).to(device)
                model = train_model(
                    model, dl_train, dl_val,
                    lambda_ss=lambda_ss, lambda_tr=lambda_tr,
                    device=device
                )

                yh_val, y_val = predict(model, dl_val, device)
                yh_val = mu_y + std_y * yh_val
                y_val  = mu_y + std_y * y_val
                val_mse = float(np.mean((yh_val - y_val) ** 2))

                # --------------- checkpoint --------------------------
                study_name = trial.study.study_name        # "lstm_arch"
                ckpt_path  = CKPT_DIR / f"{study_name}_best.pt"

                # NB: study.best_value è None finché non esiste un best
                if (trial.number == 0) or (val_mse < trial.study.best_value):
                    torch.save(model.state_dict(), ckpt_path)
                    trial.set_user_attr("ckpt_path", str(ckpt_path))

                return float(np.mean((yh_val - y_val) ** 2))

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); gc.collect()
                    raise optuna.TrialPruned()
                raise
            finally:
                del model
                torch.cuda.empty_cache(); gc.collect()

        study_lambda_ss = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name="lstm_pi"
        )
        study_lambda_ss.optimize(
            objective_lambda_ss,
            n_trials=n_trials,
            callbacks=[lambda s, t: send_telegram_message(f"λ_ss trial {t.number}: {t.value:.3e}", parse_mode=None)]
        )
        best_lambda_ss = study_lambda_ss.best_params["lambda_ss"]

    # ---------------------------------------------------------------------
    # 3) λ_tr search (λ_ss fixed al migliore)
    # ---------------------------------------------------------------------
    if best_arch.get("lambda_tr") is not None:
        logger.info("λ_tr is fixed to %.3e – skipping λ_tr search", best_arch["lambda_tr"])
        best_lambda_tr = best_arch["lambda_tr"]

    else:
        logger.info("Starting λ_tr tuning (%d trials)…", n_trials)

        def objective_lambda_tr(trial: optuna.Trial):
            try:
                lambda_ss = best_lambda_ss
                lambda_tr = trial.suggest_float("lambda_tr", 1e-7, 1e-5, log=True)

                model = LSTMModel(
                    INPUT_SIZE,
                    best_arch["hidden_size"],
                    best_arch["num_layers"]
                ).to(device)
                model = train_model(
                    model, dl_train, dl_val,
                    lambda_ss=lambda_ss, lambda_tr=lambda_tr,
                    device=device
                )

                yh_val, y_val = predict(model, dl_val, device)
                yh_val = mu_y + std_y * yh_val
                y_val  = mu_y + std_y * y_val
                val_mse = float(np.mean((yh_val - y_val) ** 2))

                # --------------- checkpoint --------------------------
                study_name = trial.study.study_name        # "lstm_arch"
                ckpt_path  = CKPT_DIR / f"{study_name}_best.pt"

                # NB: study.best_value è None finché non esiste un best
                if (trial.number == 0) or (val_mse < trial.study.best_value):
                    torch.save(model.state_dict(), ckpt_path)
                    trial.set_user_attr("ckpt_path", str(ckpt_path))

                return float(np.mean((yh_val - y_val) ** 2))

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); gc.collect()
                    raise optuna.TrialPruned()
                raise
            finally:
                del model
                torch.cuda.empty_cache(); gc.collect()

        study_lambda_tr = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            study_name="lstm_lambda_tr"
        )
        study_lambda_tr.optimize(
            objective_lambda_tr,
            n_trials=n_trials,
            callbacks=[lambda s, t: send_telegram_message(f"λ_tr trial {t.number}: {t.value:.3e}", parse_mode=None)]
        )
        best_lambda_tr = study_lambda_tr.best_params["lambda_tr"]

    best_pi_params = {
        **best_arch,
        "lambda_ss": best_lambda_ss,
        "lambda_tr": best_lambda_tr
    }
    logger.info("Best PI parameters: %s", best_pi_params)

    # ---------------------------------------------------------------------
    # 4) Retrain best models (unchanged)
    # ---------------------------------------------------------------------
    def build_model_from_params(params, ckpt_tag):
        cfg = importlib.import_module("src.config")
        cfg.LEARNING_RATE = params["lr"]

        model = LSTMModel(INPUT_SIZE,
                        params["hidden_size"],
                        params["num_layers"]).to(device)

        # ---- se esiste un checkpoint salvato prima, riparti da lì
        ckpt_file = CKPT_DIR / f"{ckpt_tag}_best.pt"
        if ckpt_file.exists():
            model.load_state_dict(torch.load(ckpt_file, map_location=device))
            logger.info("Loaded checkpoint %s", ckpt_file)
        else:
            logger.info("No checkpoint %s – training from scratch", ckpt_file)

        model = train_model(
            model, dl_train, dl_val,
            lambda_ss=params["lambda_ss"],
            lambda_tr=params["lambda_tr"],
            device=device,
            warmup_epochs=0 if not search_ss and not search_tr else LAMBDA_WARMUP_EPOCHS,
        )

        final_ckpt = CKPT_DIR / f"{ckpt_tag}_final.pt"
        torch.save(model, final_ckpt)

        return model

    logger.info("Retraining best standard LSTM")
    model_std = build_model_from_params(best_arch, "lstm_std")

    logger.info("Retraining best PI-LSTM")
    model_pi  = build_model_from_params(best_pi_params, "lstm_pi")

    import json, pathlib
    with open(pathlib.Path("checkpoints/") / "best_params.json", "w") as f:
        json.dump({
            "std": best_arch,
            "pi": best_pi_params
        }, f, indent=4)

    # FLOPs + param counts (unchanged)
    if THOP_AVAILABLE:
        dummy = torch.randn(1, WINDOW_SIZE, INPUT_SIZE).to(device)
        flops_std, params_std = profile(model_std, inputs=(dummy,), verbose=False)
        flops_pi,  params_pi  = profile(model_pi,  inputs=(dummy,), verbose=False)
        logger.info("Standard LSTM : %.2e FLOPs, %d params", flops_std, params_std)
        logger.info("PI-LSTM       : %.2e FLOPs, %d params", flops_pi,  params_pi)
    else:
        logger.info("Install `thop` to measure FLOPs.")

    # ---------------------------------------------------------------------
    # 5) ODE reference (unchanged)
    # ---------------------------------------------------------------------
    logger.info("Integrating ODE on each test cycle…")
    T_ode_list, ode_time_tot = [], 0.0
    for i in idx_test:
        T_i, ode_time = solve_reference_ode(
            cycles_t[i], cycles_P[i], cycles_Tbp[i]
        )
        T_ode_list.append(T_i)
        ode_time_tot += ode_time
    T_ode = concat_with_pad(T_ode_list, PAD)
    logger.info("ODE integration completed in %.2f s", ode_time_tot)

    # ---------------------------------------------------------------------
    # 6) Evaluation on test set + MC-Dropout (unchanged)
    # ---------------------------------------------------------------------
    logger.info("Evaluating best models on the test set…")
    yhat_std_pt, y_gt = predict(model_std, dl_test, device)
    yhat_pi_pt,  _    = predict(model_pi,  dl_test, device)

    logger.info("Running MC-dropout…")
    mu_std,  sig_std  = predict_mc(model=model_std, dataloader=dl_test, n_samples=30, device=device)
    mu_pi,   sig_pi   = predict_mc(model=model_pi,  dataloader=dl_test, n_samples=30, device=device)

    for arr in (yhat_std_pt, yhat_pi_pt, mu_std, sig_std, mu_pi, sig_pi, y_gt):
        arr *= std_y
    mu_std += mu_y
    mu_pi  += mu_y
    y_gt   += mu_y
    yhat_std_pt += mu_y
    yhat_pi_pt  += mu_y

    mse_std = np.mean((mu_std - y_gt) ** 2)
    mse_pi  = np.mean((mu_pi  - y_gt) ** 2)

    cov_std = np.mean((y_gt >= mu_std - 1.96*sig_std) &
                      (y_gt <= mu_std + 1.96*sig_std))
    cov_pi  = np.mean((y_gt >= mu_pi  - 1.96*sig_pi ) &
                      (y_gt <= mu_pi  + 1.96*sig_pi ))
    
    # ----- nuovi indicatori ----------------------------------------------
    mae_std   = mae(mu_std, y_gt)
    mae_pi    = mae(mu_pi,  y_gt)

    huber_std = huber(mu_std, y_gt, delta=1.0)
    huber_pi  = huber(mu_pi,  y_gt, delta=1.0)

    nll_std   = gaussian_nll(mu_std, sig_std, y_gt)
    nll_pi    = gaussian_nll(mu_pi,  sig_pi,  y_gt)

    logger.info("==== Test-set metrics ====")
    logger.info("MSE   – Std : %.4f   | PI : %.4f", mse_std,  mse_pi )
    logger.info("MAE   – Std : %.4f   | PI : %.4f", mae_std,  mae_pi )
    logger.info("Huber – Std : %.4f   | PI : %.4f (δ=1°C)", huber_std, huber_pi)
    logger.info("NLL   – Std : %.4f   | PI : %.4f", nll_std,  nll_pi )
    logger.info("95 %% coverage – Std : %5.2f%% | PI : %5.2f%%", 100*cov_std, 100*cov_pi)

    # ------------- ODE error
    start            = WINDOW_SIZE - 1
    t_full           = concat_with_pad([cycles_t[i] for i in idx_test], PAD)[start:]
    T_gt_full        = concat_with_pad([cycles_Tjr[i] for i in idx_test], PAD)[start:]
    T_ode_full       = T_ode[start:]
    mask_full        = ~np.isnan(T_gt_full)
    t_valid          = t_full[mask_full]
    T_gt_valid       = T_gt_full[mask_full]
    T_ode_valid      = T_ode_full[mask_full]
    mse_ode          = np.mean((T_ode_valid - T_gt_valid) ** 2)

    # ------------- Time stamps for NN predictions
    t_pred_list = []
    for i in idx_test:
        t_cycle   = cycles_t[i]
        X_cycle   = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y_fin     = cycles_Tjr[i]
        t_end     = t_cycle[WINDOW_SIZE - 1:]
        X_win, y_fin = sliding_windows(X_cycle, y_fin)
        mask = np.isfinite(X_win).all(axis=(1, 2)) & np.isfinite(y_fin)
        t_pred_list.append(t_end[mask])
    t_pred = np.concatenate(t_pred_list)

    # ---------------------------------------------------------------------
    # 7) Plots
    # ---------------------------------------------------------------------
    Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

    logger.info("Generating plots…")
    # 1) Temperature comparison
    plt.figure(figsize=(10, 4))
    plt.title("Test set – Tj comparison (Optuna-tuned models)")

    plt.plot(t_valid, T_gt_valid,            label="Ground-truth Tj",  linewidth=1.5)
    plt.plot(t_valid, T_ode_valid, "--",     label="ODE (1-RC)",       linewidth=1)
    plt.plot(t_pred,  mu_std,       ":",     label="Best LSTM (μ)",    linewidth=1)
    plt.plot(t_pred,  mu_pi,        ":",     label="Best PI-LSTM (μ)", linewidth=1)

    plt.fill_between(
        t_pred,
        mu_pi - 1.96 * sig_pi,
        mu_pi + 1.96 * sig_pi,
        alpha=0.25,
        linewidth=0,
        label="95 % PI-LSTM CI"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(PLOT_PATH) / "test_results_optuna.png", dpi=PLOT_DPI)
    plt.close()

    # 2) Error curves
    plt.figure(figsize=(10, 4))
    plt.title("Error plots – ODE vs Optuna-tuned models")

    plt.plot(t_valid, T_ode_valid - T_gt_valid, label="ODE error",          linewidth=1)
    plt.plot(t_pred,  mu_std - y_gt,            label="Best LSTM error",    linewidth=1)
    plt.plot(t_pred,  mu_pi  - y_gt,            label="Best PI-LSTM error", linewidth=1)

    plt.fill_between(
        t_pred,
        (mu_pi - 1.96 * sig_pi) - y_gt,
        (mu_pi + 1.96 * sig_pi) - y_gt,
        alpha=0.25,
        linewidth=0,
        label="95 % PI-LSTM CI"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("ΔT [°C]")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(PLOT_PATH) / "error_comparison_optuna.png", dpi=PLOT_DPI)
    plt.close()

    logger.info("All done – plots saved in %s", PLOT_PATH)

    # ---------------------------------------------------------------------
    # 8) Telegram final notification
    # ---------------------------------------------------------------------
    if use_telegram:
        msg_lines = [
            "*Optuna run finished*",
            f"`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "*Best hyper-parameters*",
            f"- Hidden size   : {best_arch['hidden_size']}",
            f"- Num layers    : {best_arch['num_layers']}",
            f"- LR            : {best_arch['lr']:.3e}",
            f"- λ_ss (PI)     : {best_lambda_ss:.3e}",
            f"- λ_tr (PI)     : {best_lambda_tr:.3e}",
            "",
            "*Test metrics*",
            f"- ODE MSE        : {mse_ode:.4f}",
            f"- MSE (Std)      : {mse_std:.4f}",
            f"- MSE (PI)       : {mse_pi:.4f}",
            f"- MAE (Std)      : {mae_std:.4f}",
            f"- MAE (PI)       : {mae_pi:.4f}",
            f"- Huber (Std)    : {huber_std:.4f}",
            f"- Huber (PI)     : {huber_pi:.4f}",
            f"- NLL (Std)      : {nll_std:.4f}",
            f"- NLL (PI)       : {nll_pi:.4f}",
            f"- 95% Cov (Std)  : {100*cov_std:.2f}",
            f"- 95% Cov (PI)   : {100*cov_pi:.2f}",
            "",
            "Plots saved in `plots/`."
        ]
        send_telegram_message("\n".join(msg_lines), parse_mode="Markdown")


# ---------------------------------------------------------------------
# 9. Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="LSTM training with Optuna – flexible search")

    parser.add_argument("--trials", type=int, default=30, help="Number of Optuna trials per study (default: 30)")
    parser.add_argument("--verbose", action="store_true", help="Also print log messages to stdout")
    parser.add_argument("--device", type=str, default=default_device, choices=["cpu", "cuda"], help="Device to use for training (default: 'cuda' if available, else 'cpu')")
    parser.add_argument("--amp", action="store_true", default=False, help="Use automatic mixed precision (default: False)")
    parser.add_argument("--use-telegram", action="store_true", default=False, help="Send notifications to Telegram (default: False)")

    # --- new search flags --------------------------------------------------
    parser.add_argument("--no-hidden-search", action="store_true", help="Do NOT search hidden_size – keep it fixed")
    parser.add_argument("--no-layers-search", action="store_true", help="Do NOT search num_layers – keep it fixed")
    parser.add_argument("--no-lr-search", action="store_true", help="Do NOT search learning-rate – keep it fixed")
    parser.add_argument("--no-ss-search", action="store_true", help="Do NOT search λ_ss – keep it fixed")
    parser.add_argument("--no-tr-search", action="store_true", help="Do NOT search λ_tr – keep it fixed")

    # Values to use when search is disabled
    parser.add_argument("--hidden-size", type=int, default=16, help="Fixed hidden_size if search disabled (default: 16)")
    parser.add_argument("--num-layers", type=int, default=1, help="Fixed num_layers if search disabled (default: 1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Fixed learning rate if search disabled (default: config value)")
    parser.add_argument("--lambda-ss", type=float, default=9.7800629709722e-06, help="Fissa λ_ss e salta la sua ricerca se non è None")
    parser.add_argument("--lambda-tr", type=float, default=2.1549713272239936e-06, help="Fissa λ_tr e salta la sua ricerca se non è None")

    args = parser.parse_args()

    #python optuna_run.py --trials 50 --verbose --use-telegram --no-hidden-search --no-layers-search --no-lr-search --no-ss-search --no-tr-search

    main(
        n_trials=args.trials,
        verbose=args.verbose,
        device=args.device,
        use_amp=args.amp,
        use_telegram=args.use_telegram,
        search_hidden=not args.no_hidden_search,
        search_layers=not args.no_layers_search,
        search_lr=not args.no_lr_search,
        search_ss=not args.no_ss_search,
        search_tr=not args.no_tr_search,
        fixed_hidden=args.hidden_size,
        fixed_layers=args.num_layers,
        fixed_lr=args.lr,
        fixed_ss=args.lambda_ss,
        fixed_tr=args.lambda_tr
    )
