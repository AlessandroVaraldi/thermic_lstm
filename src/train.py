"""
Training loop shared by both standard and physics-informed models
with optional mixed-precision (AMP) support.
"""

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast 

from src.phys_models import T_steady_state, transient_residual
from src.config import (
    PLOT_DPI, MAX_EPOCHS, PATIENCE, LEARNING_RATE,
    LAMBDA_SS, LAMBDA_TR, RTH_C, RTH_V, C_TH, T_ENV, DT
)


def train_model(
    model,
    train_dl,
    val_dl,
    lambda_ss: float = LAMBDA_SS,
    lambda_tr: float = LAMBDA_TR,
    device: str = "cpu",
    use_amp: bool = True,
    logger: logging.Logger | None = None
):
    """
    Generic trainer for both data-driven (lambda_phys = 0)
    and physics-informed (lambda_phys > 0) settings.
    Early-stops on validation loss and returns the best weights.

    Mixed precision (AMP) is enabled automatically when `device`
    is a CUDA device and a GPU is available.
    """
    # ------------------------------------------------------------------
    # Setup: loss, optimiser, AMP
    # ------------------------------------------------------------------
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler  = GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_w   = deepcopy(model.state_dict())
    patience = 0

    train_hist: list[float] = []
    val_hist:   list[float] = []

    log = logger or logging.getLogger(__name__)
    log.info("Starting training with AMP=%s (Ctrl-C to interrupt)", use_amp)

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    try:
        for epoch in range(1, MAX_EPOCHS + 1):

            # ==================== TRAIN ================================
            model.train()
            batch_losses: list[float] = []

            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)

                # Forward pass under autocast
                with autocast(device_type="cuda", enabled=use_amp):
                    y_pred = model(xb)
                    loss   = criterion(y_pred, yb)

                    # ---- physics consistency terms -------------------
                    P_c, Tbp_c = xb[:, -1, 0], xb[:, -1, 1]
                    P_p, Tbp_p = xb[:, -2, 0], xb[:, -2, 1]

                    loss_ss = 0.0
                    if lambda_ss > 0:
                        T_ss   = T_steady_state(P_c, Tbp_c, T_ENV, RTH_C, RTH_V)
                        loss_ss = criterion(y_pred, T_ss)

                    xb_prev = torch.cat([xb[:, :1, :], xb[:, :-1, :]], 1)
                    T_prev  = model(xb_prev)

                    res_tr  = transient_residual(
                        T_prev, y_pred,
                        P_p, P_c,
                        Tbp_p, Tbp_c,
                        DT, RTH_C, RTH_V, C_TH, T_ENV
                    )
                    loss_tr = torch.mean(res_tr ** 2)

                    loss = loss + lambda_ss * loss_ss + loss_tr * lambda_tr

                # ---- backward + optimisation ------------------------
                optimiser.zero_grad(set_to_none=True)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimiser)                     # unscale for clipping
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()

                batch_losses.append(loss.item())

            avg_train = sum(batch_losses) / len(batch_losses)
            train_hist.append(avg_train)

            # ==================== VALIDATION ===========================
            model.eval()
            val_losses = []

            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(device), yb.to(device)

                    with autocast(device_type="cuda", enabled=use_amp):
                        y_pred = model(xb)
                        loss   = criterion(y_pred, yb)

                        if lambda_ss > 0:
                            P_c, Tbp_c = xb[:, -1, 0], xb[:, -1, 1]
                            P_p, Tbp_p = xb[:, -2, 0], xb[:, -2, 1]

                            T_ss   = T_steady_state(P_c, Tbp_c, T_ENV, RTH_C, RTH_V)
                            loss_ss = criterion(y_pred, T_ss)

                            xb_prev = torch.cat([xb[:, :1, :], xb[:, :-1, :]], 1)
                            T_prev  = model(xb_prev)

                            res_tr  = transient_residual(
                                T_prev, y_pred,
                                P_p, P_c,
                                Tbp_p, Tbp_c,
                                DT, RTH_C, RTH_V, C_TH, T_ENV
                            )
                            loss_tr = torch.mean(res_tr ** 2)
                            loss    = loss + lambda_ss *  loss_ss + loss_tr * lambda_tr

                    val_losses.append(loss.item())

            cur_val = sum(val_losses) / len(val_losses)
            val_hist.append(cur_val)

            log.info("[%03d] train=%.6f  val=%.6f  (best %.6f)",
                     epoch, avg_train, cur_val, best_val)

            # ==================== EARLY-STOPPING =======================
            if cur_val < best_val:
                best_val = cur_val
                best_w   = deepcopy(model.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    log.info("Early stopping triggered.")
                    break

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")

    # ------------------------------------------------------------------
    # Plot loss curves
    # ------------------------------------------------------------------
    Path("plots").mkdir(exist_ok=True)
    epochs = range(1, len(train_hist) + 1)

    plt.figure()
    plt.plot(epochs, train_hist, label="Train")
    plt.plot(epochs, val_hist,   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(
        "Physics-Informed Training" if lambda_ss > 0 else "Data-Driven Training"
    )
    plt.legend(); plt.grid(True); plt.tight_layout()

    fname = ("plots/phys_informed_loss_curves.png"
             if lambda_ss > 0 else
             "plots/data_driven_loss_curves.png")
    plt.savefig(fname, dpi=PLOT_DPI)
    plt.close()

    log.info("Training completed. Best val loss = %.6f", best_val)

    # ------------------------------------------------------------------
    # Return best model
    # ------------------------------------------------------------------
    model.load_state_dict(best_w)
    return model
