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

# Resolution of the plots saved in `PLOT_PATH`. The resolution is
# specified as a dpi value, which is the number of pixels per inch.
PLOT_DPI = 3000  # int

# --------------------------------------------------------------------
# Data-set parameters
# --------------------------------------------------------------------
# Length (in time-steps) of each sliding window fed to the LSTM. A
# window of 128 samples roughly corresponds to ~12 seconds of data.
WINDOW_SIZE = 128  # int

# Fractions used to split *time-ordered* data into train/val/test. The
# remaining 10 % (1 - TRAIN_RATIO - VAL_RATIO) is implicitly the test
# share. Ratios are applied after optional augmentations.
TRAIN_RATIO = 0.70  # float in [0,1]
VAL_RATIO   = 0.20  # float in [0,1]

# --------------------------------------------------------------------
# Model architecture parameters
# --------------------------------------------------------------------
# Number of features per time-step: here we feed net power (P_net) and
# base-plate temperature (Tbp) → 2 features in total.
INPUT_SIZE  = 2   # int

# Hidden state size of the LSTM layer. Larger values boost model
# capacity at the cost of more parameters and slower inference.
HIDDEN_SIZE = 16  # int

# Number of features in the output layer. We predict a single scalar
# value (junction temperature) at each time-step, so this is 1.
OUTPUT_SIZE = 1   # int

# How many stacked LSTM layers to use.
NUM_LAYERS  = 1   # int

# Scaling factor applied to the physics-informed penalties (steady-state
# loss).
LAMBDA_SS = 1e-5  # float ≥ 0
# Scaling factor applied to the thermal resistance loss.
LAMBDA_TR = 1e-5  # float ≥ 0

# --------------------------------------------------------------------
# NEW: scheduler warm-up
# --------------------------------------------------------------------
# Number of epochs over which λ_ss and λ_tr ramp up linearly
# from 0 → λ_max. 0 = scheduler disabilitato (comportamento attuale).
LAMBDA_WARMUP_EPOCHS = 10  # int ≥ 0

# --------------------------------------------------------------------
# Training hyper-parameters
# --------------------------------------------------------------------
SEED          = 97     # int – global random seed for reproducibility
MAX_EPOCHS    = 1000   # int – upper bound on training epochs
PATIENCE      = 10     # int – early-stopping patience on validation loss
BATCH_SIZE    = 8      # int – number of windows per mini-batch
LEARNING_RATE = 1e-4   # float – initial LR for the Adam optimiser

# --------------------------------------------------------------------
# Physical constants
# --------------------------------------------------------------------
# Thermal resistance [K/W] from junction → case.
RTH_C = 1.65      # float (Kelvin per Watt)
# Thermal resistance [K/W] from junction → ambient.
RTH_V = 1e5       # float (Kelvin per Watt)
# Lumped thermal capacitance [J/K] of the chip package.
C_TH  = 1.5       # float (Joule per Kelvin)
# Ambient (coolant) temperature [°C].
T_ENV = 20.0      # float (degree Celsius)
# Drain-source on-state resistance of the MOSFET [Ω]
R_DSON = 0.63e-3  # float (Ohm)

# --------------------------------------------------------------------
# Sampling period (will be overwritten at runtime by optuna_run.py)
# --------------------------------------------------------------------
DT = 1e-2         # float (seconds)

# --------------------------------------------------------------------
# Data-augmentation parameters (cycle-level)
# --------------------------------------------------------------------
# Number of *additional* synthetic duty-cycles generated via simple
# noise & scaling tricks defined in `data_utils.augment_cycle()`.
AUG_CYCLES = 5   # int ≥ 0

# Fractions used when splitting *cycles* (as opposed to individual
# samples) into train/val/test datasets after augmentation.
TRAIN_FRAC = 0.70 # float in [0,1]
VAL_FRAC   = 0.15 # float in [0,1]

# Standard deviation (relative) of the additive Gaussian noise applied
# to temperature channels during augmentation.
NOISE_STD = 0.02  # float ≥ 0

# Standard deviation (relative) of the multiplicative scaling applied
# to power traces during augmentation (≈ ±5 %).
SCALE_STD = 0.05  # float ≥ 0

# Varianza (°C) dell’offset termico quasi-statico che simula
# cambiamenti lentamente varianti di ambient/cooling.
TEMP_OFFSET_STD = 2.0      # float ≥ 0  (≈ ±2 °C 1 σ)

# Massimo jitter temporale (campioni) applicato al ciclo completo
# (P, Tbp, Tjr) per emulare variazioni di fase / latenze di misura.
JITTER_SAMPLES  = 5        # int ≥ 0    (~5 × DT ≈ 50 ms con dt=0.01 s)

# --------------------------------------------------------------------
# End of configuration – nothing below this line should normally be
# edited. All other modules import from this centralised file.
# --------------------------------------------------------------------
