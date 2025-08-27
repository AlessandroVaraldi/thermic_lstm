"""
Physical helper functions (ODE and steady-state temperature).
Keep all physics-based utilities in one place.
"""

import numpy as np

def thermal_ode(T, t, t_data, P_data, Tbp_data, Tenv_data, RthC, RthV, Cth):
    """Single-state thermal RC ODE."""
    P_val    = np.interp(t, t_data, P_data)
    Tbp_val  = np.interp(t, t_data, Tbp_data)
    Tenv_val = np.interp(t, t_data, Tenv_data)

    dTdt = (P_val + (Tbp_val - T)/RthC + (Tenv_val - T)/RthV) / Cth
    return dTdt


def T_steady_state(P, Tbp, Tenv, RthC, RthV):
    """
    Analytical steady-state solution of thermal_ode.
    Handy for the physics-informed loss term.
    """
    denom = 1.0/RthC + 1.0/RthV
    numer = P + Tbp/RthC + Tenv/RthV
    return numer / denom


# ---------------------------------------------------------------------------
# Transient residual – first-order energy balance
# ---------------------------------------------------------------------------
def transient_residual(T_prev, T_curr,
                       P_prev, P_curr,
                       Tbp_prev, Tbp_curr,
                       dt, RthC, RthV, Cth, Tenv):
    """
    Residual of the single-RC ODE evaluated over one time-step.

    Parameters
    ----------
    T_prev, T_curr : torch.Tensor
        Predicted junction temperature at *t-Δt* and *t*  (shape: B).
    P_prev, P_curr : torch.Tensor
        Net power at the two instants                        (shape: B).
    Tbp_prev, Tbp_curr : torch.Tensor
        Base-plate temperature at the two instants           (shape: B).
    dt : float
        Sampling period [s].
    RthC, RthV, Cth, Tenv : float
        Physical parameters of the thermal network.

    Returns
    -------
    torch.Tensor
        Residual r = dT/dt − f(T, P, Tbp)   (shape: B).
    """
    import torch  # local import avoids cyclic deps

    dTdt_pred = (T_curr - T_prev) / dt
    rhs = (P_curr
           + (Tbp_curr - T_curr)/RthC
           + (Tenv     - T_curr)/RthV) / Cth
    return dTdt_pred - rhs
