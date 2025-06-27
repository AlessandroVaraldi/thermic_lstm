# ---------------------------------------------------------------------
# 1. Standard libraries
# ---------------------------------------------------------------------
import argparse
import logging
import os
import time
from pathlib import Path

# ---------------------------------------------------------------------
# 2. Third-party libraries
# ---------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import ConcatDataset, DataLoader

import requests  # ← new

try:
    import optuna
except ImportError as e:
    raise ImportError("Optuna not found - install with `pip install optuna`") from e

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# ---------------------------------------------------------------------
# 3. Local modules
# ---------------------------------------------------------------------
from src.config import *                     # global project settings
from src.data_utils import (
    load_csv, compute_powers, augment_cycle, seed_everything,
    solve_reference_ode, sliding_windows, predict_mc
)
from src.dataset import WindowDataset
from src.models import LSTMModel
from src.train import train_model

# ---------------------------------------------------------------------
# 4. Logging helpers
# ---------------------------------------------------------------------
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

    logger = logging.getLogger()          # root logger
    logger.setLevel(logging.INFO)

    # -------- NEW: remove any handler that was already attached
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
        ch = logging.StreamHandler()      # single console handler
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.info("Logger initialised - log file: %s", log_path)
    return logger


# ---------------------------------------------------------------------
# 5. Utility functions (unchanged except for logging)
# ---------------------------------------------------------------------
def predict(model, dl, device):
    """Forward pass over a DataLoader - returns stacked numpy arrays."""
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds.append(model(xb.to(device)).cpu().numpy())
            gts.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(gts)


def concat_with_pad(arr_list, pad_len):
    """Concatenate an iterable of 1-D arrays inserting NaN padding."""
    if len(arr_list) == 0:
        return np.empty(0, dtype=float)
    pad = np.full(pad_len, np.nan, dtype=arr_list[0].dtype)
    return np.concatenate([np.concatenate([a, pad]) for a in arr_list])[:-pad_len]


def build_concat_dataset(
    idxs, cycles_P, cycles_Tbp, cycles_Tjr, mu_x, std_x, mu_y, std_y
):
    """Builds a torch.utils.data.ConcatDataset out of selected cycle indices."""
    ds_list = []
    for i in idxs:
        X = np.column_stack([cycles_P[i], cycles_Tbp[i]])
        y = cycles_Tjr[i]
        ds_list.append(WindowDataset(X, y, mu_x, std_x, mu_y, std_y))
    return ConcatDataset(ds_list)


# ---------------------------------------------------------------------
# 6. Telegram helpers
# ---------------------------------------------------------------------
def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    parse_mode: str = "Markdown"
) -> bool:
    """
    Push a message to Telegram via Bot API.

    Parameters
    ----------
    text : str
        Message body (supports Markdown / HTML depending on `parse_mode`).
    bot_token : str | None
        Your bot token (fallback to env var TELEGRAM_BOT_TOKEN).
    chat_id : str | None
        Destination chat ID (fallback to env var TELEGRAM_CHAT_ID).
    parse_mode : str
        'Markdown', 'MarkdownV2' or 'HTML'.

    Returns
    -------
    bool
        True on success, False otherwise (errors are logged).
    """
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = chat_id   or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        logging.warning("Telegram token / chat_id missing - notification skipped")
        return False

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

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
def main(n_trials: int, verbose: bool, device: str = "cuda", use_amp: bool = True, use_telegram: bool = True):

    if use_telegram:
        send_telegram_message(
            f"Starting LSTM training with {n_trials} trials",
            parse_mode="MarkdownV2"
        )

    logger = setup_logging(verbose)

    # -------------------------------------------------- Reproducibility
    seed_everything(SEED)
    logger.info("Random seed set to %s", SEED)

    # -------------------------------------------------- Data loading
    cols   = load_csv()
    P_orig = compute_powers(cols["Id"], cols["Iq"])

    t_base     = cols["t"]
    dt         = t_base[1] - t_base[0]
    # expose dt to the training loop via the global config
    import importlib; importlib.import_module("src.config").DT = float(dt)
    cycle_span = t_base[-1] - t_base[0]
    logger.info("Loaded dataset - base cycle length: %.3f s, dt: %.6f s", cycle_span, dt)

    # ----------- Original + augmented duty-cycles
    cycles_t, cycles_P, cycles_Tbp, cycles_Tjr = [], [], [], []
    cycles_t  .append(t_base.copy())
    cycles_P  .append(P_orig.copy())
    cycles_Tbp.append(cols["Tbp"].copy())
    cycles_Tjr.append(cols["Tjr"].copy())

    for i in range(AUG_CYCLES):
        P_aug, Tbp_aug, Tjr_aug = augment_cycle(P_orig, cols["Tbp"], cols["Tjr"])
        t_shift = (i + 1) * (cycle_span + dt)
        cycles_t  .append(t_base + t_shift)
        cycles_P  .append(P_aug)
        cycles_Tbp.append(Tbp_aug)
        cycles_Tjr.append(Tjr_aug)
    logger.info("Augmented %d additional cycles (total = %d)", AUG_CYCLES, len(cycles_t))

    n_cycles = len(cycles_t)
    n_train  = int(TRAIN_FRAC * n_cycles)
    n_val    = int(VAL_FRAC   * n_cycles)

    idx_train = list(range(0,               n_train))
    idx_val   = list(range(n_train,         n_train + n_val))
    idx_test  = list(range(n_train + n_val, n_cycles))

    PAD = WINDOW_SIZE - 1  # pad between cycles when concatenating

    # ----------- Flattened (train/val/test) signals - needed for μ/σ
    P_train  = concat_with_pad([cycles_P[i]   for i in idx_train], PAD)
    Tbp_train = concat_with_pad([cycles_Tbp[i] for i in idx_train], PAD)

    X_train_raw = np.column_stack([P_train, Tbp_train])
    mu_x, std_x = np.nanmean(X_train_raw, axis=0), np.nanstd(X_train_raw, axis=0)

    Tjr_train = concat_with_pad([cycles_Tjr[i] for i in idx_train], PAD)
    mu_y, std_y = np.nanmean(Tjr_train), np.nanstd(Tjr_train)
    logger.info("Computed normalisation statistics")

    # ----------- Datasets / DataLoaders
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

    # -------------------------------------------------- Optuna studies
    # 1) Architecture + learning-rate search
    logger.info("Starting architecture + LR tuning (%d trials)…", n_trials)

    def objective_arch(trial: optuna.Trial):
        hidden_size = trial.suggest_int("hidden_size", 8, 32, step=8)
        num_layers  = trial.suggest_int("num_layers", 1, 3)
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lambda_ss = 0.0  # fixed during first phase
        lambda_tr = 0.0  # fixed during first phase

        import importlib
        importlib.import_module("src.config").LEARNING_RATE = lr

        model = LSTMModel(INPUT_SIZE, hidden_size, num_layers).to(device)
        model = train_model(model, dl_train, dl_val, lambda_ss=lambda_ss, lambda_tr=lambda_tr, device=device, use_amp=use_amp, logger=logger)

        yh_val, y_val = predict(model, dl_val, device)
        yh_val = mu_y + std_y * yh_val
        y_val  = mu_y + std_y * y_val
        return np.mean((yh_val - y_val) ** 2)

    study_arch = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name="lstm_arch"
    )
    study_arch.optimize(objective_arch, n_trials=n_trials, callbacks=[
        lambda study, trial: send_telegram_message(
            f"Trial {trial.number} completed: {trial.value:.3e}",
            parse_mode=None
        )
    ])

    best_arch = study_arch.best_params
    best_arch["lambda_ss"] = 0.0
    best_arch["lambda_tr"] = 0.0
    logger.info("Best architecture: %s", best_arch)

    # 2) Lambda_phys search
    logger.info("Starting lambda tuning (%d trials)…", n_trials)

    import gc
    def objective_lambda(trial):
        try:
            lambda_ss = trial.suggest_float("lambda_ss", 1e-6, 1e-2, log=True)
            lambda_tr = trial.suggest_float("lambda_tr", 1e-6, 1e-2, log=True)

            model = LSTMModel(INPUT_SIZE,
                            best_arch["hidden_size"],
                            best_arch["num_layers"]).to(device)
            model = train_model(model, dl_train, dl_val,
                                lambda_ss=lambda_ss,
                                lambda_tr=lambda_tr,
                                device=device)

            yh_val, y_val = predict(model, dl_val, device)
            yh_val = mu_y + std_y * yh_val
            y_val  = mu_y + std_y * y_val
            return float(np.mean((yh_val - y_val) ** 2))

        except RuntimeError as e:
            # graceful handling of CUDA OOM
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()    # segnala ad Optuna di scartare il trial
            raise
        finally:
            del model
            torch.cuda.empty_cache()
            gc.collect()

    study_lambda = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        study_name="lstm_lambda"
    )
    study_lambda.optimize(objective_lambda, n_trials=n_trials, callbacks=[
        lambda study, trial: send_telegram_message(
            f"Trial {trial.number} completed: {trial.value:.3e}",
            parse_mode=None
        )
    ])

    best_pi_params = {**best_arch, "lambda_ss": study_lambda.best_params["lambda_ss"], "lambda_tr": study_lambda.best_params["lambda_tr"]}
    logger.info("Best lambda parameters: %s", best_pi_params)

    # -------------------------------------------------- Retrain best models
    def build_model_from_params(params):
        import importlib
        cfg = importlib.import_module("src.config")
        cfg.LEARNING_RATE = params["lr"]

        model = LSTMModel(
            INPUT_SIZE,
            params["hidden_size"],
            params["num_layers"]
        ).to(device)

        return train_model(
            model, dl_train, dl_val,
            lambda_ss=params["lambda_ss"],
            lambda_tr=params["lambda_tr"],
            device=device
        )

    logger.info("Retraining best standard LSTM")
    model_std = build_model_from_params(best_arch)

    logger.info("Retraining best PI-LSTM")
    model_pi  = build_model_from_params(best_pi_params)

    # FLOPs + param counts
    if THOP_AVAILABLE:
        dummy = torch.randn(1, WINDOW_SIZE, INPUT_SIZE).to(device)
        flops_std, params_std = profile(model_std, inputs=(dummy,), verbose=False)
        flops_pi,  params_pi  = profile(model_pi,  inputs=(dummy,), verbose=False)
        logger.info("Standard LSTM : %.2e FLOPs, %d params", flops_std, params_std)
        logger.info("PI-LSTM       : %.2e FLOPs, %d params", flops_pi,  params_pi)
    else:
        logger.info("Install `thop` to measure FLOPs.")

    # -------------------------------------------------- ODE reference (test set only)
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

    # -------------------------------------------------- Evaluation on test set
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

    logger.info("MSE - Standard LSTM : %.4f", mse_std)
    logger.info("MSE - PI-LSTM       : %.4f", mse_pi)
    logger.info("95 %% coverage - Std : %5.2f%%", 100*cov_std)
    logger.info("95 %% coverage - PI  : %5.2f%%", 100*cov_pi)

    # ------------- ODE error (aligned to ground truth & masked)
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

    # -------------------------------------------------- Plots
    Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

    logger.info("Generating plots…")
    # 1) Temperature comparison
    plt.figure(figsize=(10, 4))
    plt.title("Test set - Tj comparison (Optuna-tuned models)")

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
    plt.title("Error plots - ODE vs Optuna-tuned models")

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

    logger.info("All done - plots saved in %s", PLOT_PATH)

    # -------------------------------------------------- Telegram notification
    if use_telegram:
        msg_lines = [
            "*Optuna run finished*",
            f"`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "*Best hyper-parameters*",
            f"- Hidden size   : {best_arch['hidden_size']}",
            f"- Num layers    : {best_arch['num_layers']}",
            f"- LR            : {best_arch['lr']:.3e}",
            f"- λ_ss (PI)     : {best_pi_params['lambda_ss']:.3e}",
            f"- λ_tr (PI)     : {best_pi_params['lambda_tr']:.3e}",
            "",
            "*Test metrics*",
            f"- MSE LSTM      : {mse_std:.4f}",
            f"- MSE PI-LSTM   : {mse_pi:.4f}",
            f"- 95 % cov LSTM : {cov_std*100:.2f} %",
            f"- 95 % cov PI   : {cov_pi*100:.2f} %",
            "",
            "Plots saved in `plots/`."
        ]
        send_telegram_message("\n".join(msg_lines), parse_mode="Markdown")


# ---------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description="LSTM training with Optuna")
    parser.add_argument(
        "--trials", type=int, default=30,
        help="Number of Optuna trials per study (default: 30)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Also print log messages to stdout"
    )
    parser.add_argument(
        "--device", type=str, 
        default=default_device,
        choices=["cpu", "cuda"],
        help="Device to use for training (default: 'cuda' if available, else 'cpu')"
    )
    parser.add_argument(
        "--amp", action="store_true",
        default=True,
        help="Use automatic mixed precision (AMP) for training (default: True)"
    )
    parser.add_argument(
        "--use-telegram", action="store_true",
        default=False,
        help="Send notifications to Telegram (default: True)"
    )
   
    args = parser.parse_args()
    main(n_trials=args.trials, verbose=args.verbose, device=args.device, use_amp=args.amp, use_telegram=args.use_telegram)

