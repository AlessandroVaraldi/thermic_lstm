"""
Global configuration and hyper-parameters used throughout the project.

Every module imports *only* from this file, so updating a value here
will automatically propagate everywhere.
"""

# --------------------------------------------------------------------
# File paths
# --------------------------------------------------------------------
# Directory that contains one or più CSV omogenei fra loro
CSV_DIR  = "data_sets/"               # str
# (facoltativo) pattern glob se volete filtrare
CSV_GLOB = "*.csv"                    # str

# Directory where PNG plots (e.g. training curves, inference results)
# will be stored. The directory will be created if it does not exist.
PLOT_PATH = "plots/"  # str
# Alias used by some modules
PLOTS_DIR = PLOT_PATH  # str

# Resolution of the plots saved in `PLOT_PATH`/`PLOTS_DIR`
PLOT_DPI = 300  # int

# --------------------------------------------------------------------
# Data-set parameters
# --------------------------------------------------------------------
# Length (in time-steps) of each sliding window fed to the LSTM.
WINDOW_SIZE = 64  # int

# Fractions used to split *time-ordered* data into train/val/test.
TRAIN_RATIO = 0.70  # float in [0,1]
VAL_RATIO   = 0.20  # float in [0,1]

# Fractions used when splitting *cycles* after augmentation.
TRAIN_FRAC = 0.70   # float in [0,1]
VAL_FRAC   = 0.15   # float in [0,1]

# --------------------------------------------------------------------
# Model architecture parameters
# --------------------------------------------------------------------
# Input features per time-step (P_net, Tbp)
INPUT_SIZE  = 2    # int
# LSTM hidden size
HIDDEN_SIZE = 16   # int
# Output features (we predict a single scalar)
OUTPUT_SIZE = 1    # int
# Stacked LSTM layers
NUM_LAYERS  = 1    # int
# Dropout probability inside the LSTM/heads
DROPOUT     = 0.10 # float in [0,1]

# --------------------------------------------------------------------
# Physics-informed losses
# --------------------------------------------------------------------
# Steady-state loss weight
LAMBDA_SS = 9.78e-6  # float ≥ 0
# Transient residual loss weight
LAMBDA_TR = 2.15e-6  # float ≥ 0

# Warm-up (λ ramp-up from 0 → target)
LAMBDA_WARMUP_EPOCHS = 10  # int ≥ 0 (0 disables ramp-up)

# --------------------------------------------------------------------
# Training hyper-parameters
# --------------------------------------------------------------------
SEED          = 97     # int
MAX_EPOCHS    = 1000   # int
EPOCHS        = 100    # legacy, may be unused
PATIENCE      = 10     # int
BATCH_SIZE    = 16     # int
LEARNING_RATE = 1e-3   # float

# --------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------
RTH_C = 1.65      # K/W (junction→case)
RTH_V = 1e5       # K/W (junction→ambient)
C_TH  = 1.5       # J/K  (lumped capacitance)
T_ENV = 20.0      # °C
R_DSON = 0.63e-3  # Ω

# --------------------------------------------------------------------
# Sampling period (can be overwritten at runtime)
# --------------------------------------------------------------------
DT = 1e-2  # seconds

# --------------------------------------------------------------------
# Data-augmentation parameters (cycle-level)
# --------------------------------------------------------------------
AUG_CYCLES = 2    # int ≥ 0
NOISE_STD  = 0.02 # relative std for temperature noise
SCALE_STD  = 0.05 # relative std for power scaling
TEMP_OFFSET_STD = 2.0  # °C std of quasi-static offset
JITTER_SAMPLES  = 5    # max temporal jitter (samples)

# --------------------------------------------------------------------
# Training defaults (moved here from script)
# --------------------------------------------------------------------
# Epoch gate for enabling best/early-stop
MIN_EPOCHS_BEST = 25

# ReduceLROnPlateau scheduler settings
LR_SCHED_PLATEAU = True
LR_FACTOR        = 0.5
LR_PATIENCE      = 4
LR_MIN           = 1e-6

# QAT observer tuning
EMA_DECAY_QAT    = 0.99
Q_DELAY_UPDATES  = 400
Q_FREEZE_UPDATES = 4000

# --------------------------------------------------------------------
# Runtime controls (were CLI flags; now centralized here)
# --------------------------------------------------------------------
# Device selection
try:
    import torch as _torch
    DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

# AMP (mixed precision)
AMP_ENABLED = True           # bool
AMP_DTYPE   = "bf16"         # "bf16" or "fp16"

# Activation checkpointing / TBPTT
CKPT        = False          # bool (coarse checkpointing fallback)
CKPT_CHUNK  = 16             # int (used by fine-grained ckpt if model supports)
TBPTT_K     = 0              # int (0=disabled)

# Optimization
ACCUM       = 2              # gradient accumulation steps
FUSED_ADAM  = True           # try fused Adam if available
COMPILE     = 0              # torch.compile (0/1)

# Validation cadence
VAL_INTERVAL     = 1         # validate every K epochs
VAL_MAX_BATCHES  = 0         # 0 = no limit

# DataLoader settings
import os as _os
WORKERS   = min(8, max(1, (_os.cpu_count() or 2)//2))  # sensible default
PIN_MEMORY = True
PERSIST    = True
PREFETCH   = 4

# Temporal mixed-precision (time-aware scaling only)
MP_TIME         = 1          # 0/1
MP_TAU_THR      = 0.08       # °C/s threshold
MP_SCALE_MUL    = 1.5        # scale factor for S_gate in mp-time
MP_RSHIFT_DELTA = -1         # delta rshift for S_gate in mp-time

# --------------------------------------------------------------------
# End of configuration – nothing below this line should normally be
# edited. All other modules import from this centralised file.
# --------------------------------------------------------------------
