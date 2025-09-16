# Configuration Reference (Deep but Practical)

This file explains **every parameter** in `src/config.py` in plain language. It is meant for readers **not familiar with deep learning**. No long introductions—each item tells you **what it is**, **why it matters**, **safe ranges**, **when to change it**, and **common mistakes**.

> Notation: **Type** = Python type; **Units** where relevant; **Default** shows the current value; **Used by** = important places in the code that rely on it.

---

## 0) Problem Formulation

- Data are time series: **power** `P` and **base‑plate temperature** `Tbp`.  
- The model predicts **junction temperature** `Tjr` at each time step.
- We train on **windows** (short slices) of consecutive samples.  
- Training tries to fit data **and** respect simple physics (steady‑state and transient behavior).

---

## 1) Paths & Output

### `CSV_DIR` *(str, Default: `"data_sets/"`)*
- **What:** Folder containing input CSV files (your recordings).
- **Why:** The training script loads all CSVs from here.
- **Safe values:** An existing directory path.
- **When to change:** If your data live somewhere else.
- **Common mistakes:** Pointing to a folder that contains non‑CSV junk with the same extension.

### `CSV_GLOB` *(str, Default: `"*.csv"`)*
- **What:** Pattern that selects which files inside `CSV_DIR` are read.
- **Why:** Helps isolate a subset (e.g., `"drive_*.csv"`).
- **When to change:** Mixed data with different schemas.
- **Common mistakes:** Overly broad patterns that include malformed files.

### `PLOT_PATH` / `PLOTS_DIR` *(str, Default: `"plots/"`)*
- **What:** Where figures (e.g., training curves) are written.
- **Notes:** Created automatically if missing.

### `PLOT_DPI` *(int, Default: `300`)*
- **What:** Resolution for saved images.
- **When to change:** For papers, keep `300+`. For quick debugging, `100–150` is fine.
- **Trade‑off:** Higher DPI → larger files.

---

## 2) Dataset & Splits

### `WINDOW_SIZE` *(int, time‑steps, Default: `64`)*
- **What:** Number of consecutive samples the model sees at once.
- **Why:** The model uses recent history to predict the last time step.
- **Effects:** Larger values → more context, more memory, longer training.
- **Physics link:** Transient loss needs at least **2** steps.
- **Typical range:** `16–256`; start with `64`.
- **Common mistakes:** Setting it so large that the GPU/CPU runs out of memory.

### `TRAIN_RATIO`, `VAL_RATIO` *(floats; time‑ordered split; Defaults: `0.70`, `0.20`)*
- **What:** Legacy ratios for time‑ordered splits (not cycle‑based).
- **Use today:** Kept for backward compatibility. The active pipeline uses **cycle‑based** splits (below).
- **Constraint:** `TRAIN_RATIO + VAL_RATIO ≤ 1.0` (the rest is test).

### `TRAIN_FRAC`, `VAL_FRAC` *(floats; cycle‑based; Defaults: `0.70`, `0.15`)*
- **What:** Fractions to split **by driving cycles** after augmentation.
- **Why:** Keeps entire cycles intact across splits (better independence).
- **Constraint:** `TRAIN_FRAC + VAL_FRAC ≤ 1.0` (remainder is test).
- **Tuning tip:** If validation is noisy, increase `VAL_FRAC` to `0.2–0.3`.

---

## 3) Model Architecture

### `INPUT_SIZE` *(int, Default: `2`)*
- **What:** Features per time step (here: `P`, `Tbp`).
- **Change only if:** You add/remove input channels (then update your dataset code).

### `HIDDEN_SIZE` *(int, Default: `16`)*
- **What:** Internal width of the LSTM (its “memory capacity”).  
- **Larger =** usually more accurate but slower and heavier.  
- **Typical range:** `8–128`. Try doubling/halving when tuning.

### `OUTPUT_SIZE` *(int, Default: `1`)*
- **What:** How many values you predict per time step (here: 1 temperature).

### `NUM_LAYERS` *(int, Default: `1`)*
- **What:** How many LSTM layers are stacked.  
- **2+ layers:** more expressive, more compute.  
- **Start with:** `1`. Try `2` if underfitting persists.

### `DROPOUT` *(float in [0,1], Default: `0.10`)*
- **What:** Randomly drops part of the internal signals during training.
- **Why:** Reduces overfitting.
- **Typical range:** `0.05–0.30`. Avoid `>0.5`.

---

## 4) Physics‑Informed Terms

We add penalties to keep the model consistent with basic thermal physics.

### `LAMBDA_SS` *(float ≥ 0, Default: `9.78e-6`)*
- **What:** Weight of the **steady‑state** loss.  
- **Intuition:** If power and boundary temperature are stable, the junction temperature should approach `T_ss` computed from resistances.
- **Too high:** The model may ignore real data patterns.  
- **Too low:** Physics ignored; worse extrapolation.

### `LAMBDA_TR` *(float ≥ 0, Default: `2.15e-6`)*
- **What:** Weight of the **transient** (dynamic) residual.
- **Needs:** `WINDOW_SIZE ≥ 2`.
- **Tuning:** Start small; increase if predictions look physically implausible during transients.

### `LAMBDA_WARMUP_EPOCHS` *(int ≥ 0, Default: `10`)*
- **What:** Linearly ramps `λ_ss` and `λ_tr` from 0 to their targets over N epochs.
- **Why:** Avoids overpowering the model with physics before it learns basics from data.
- **Set to 0:** Disables ramping.

---

## 5) Training Hyper‑parameters

### `SEED` *(int, Default: `97`)*
- **What:** Reproducibility for shuffles and initializations.

### `MAX_EPOCHS` *(int, Default: `1000`)*
- **What:** Hard cap on training epochs.

### `EPOCHS` *(int, Default: `100`, legacy)*
- **What:** Legacy; not used by the current loop (kept for compatibility).

### `PATIENCE` *(int, Default: `10`)*
- **What:** Allow this many **validation checks** without improvement before early stop (after warm‑up gate).
- **Tip:** Combine with `VAL_INTERVAL` (e.g., validate every 1–3 epochs).

### `BATCH_SIZE` *(int, Default: `8`)*
- **What:** Windows per gradient step.  
- **Memory:** Larger batch needs more RAM/VRAM.  
- **Effective batch:** `BATCH_SIZE × ACCUM` (see `ACCUM`).

### `LEARNING_RATE` *(float, Default: `1e-4`)*
- **What:** Step size for the optimizer.  
- **Tuning:** Try `3e-4`, `1e-4`, `3e-5` when searching. Too large → unstable; too small → slow.

---

## 6) Physical Constants (Thermal Model)

These define a simple equivalent thermal network used in the physics terms.

### `RTH_C` *(K/W, Default: `1.65`)* — Junction→case resistance  
### `RTH_V` *(K/W, Default: `1e5`)* — Junction→ambient resistance  
### `C_TH`  *(J/K, Default: `1.5`)* — Lumped thermal capacitance  
### `T_ENV` *(°C,  Default: `20.0`)* — Ambient/coolant temperature  
### `R_DSON`*(Ω,   Default: `0.00063`)* — On‑state resistance for power model

- **Notes:** Orders of magnitude matter more than perfect accuracy.
- **When to change:** Different hardware or boundary conditions.

### `DT` *(seconds, Default: `0.01`)*
- **What:** Sampling period.  
- **Auto‑detection:** The script computes a median `dt` from the CSVs and warns if not uniform.  
- **Tip:** Keep `DT` consistent with your sensors/logging setup.

---

## 7) Data Augmentation (cycle‑level)

These make training more robust by slightly altering the original cycles.

### `AUG_CYCLES` *(int ≥ 0, Default: `2`)*
- **What:** Number of synthetic cycles generated from each original cycle.
- **Effect:** More data; longer training. Use `0–3` unless data are very scarce.

### `NOISE_STD` *(float ≥ 0, Default: `0.02`)*
- **What:** Relative Gaussian noise added to temperature channels.

### `SCALE_STD` *(float ≥ 0, Default: `0.05`)*
- **What:** Relative scaling applied to power traces (≈ ±5%).

### `TEMP_OFFSET_STD` *(°C, Default: `2.0`)*
- **What:** Slowly varying offset added to temperatures to mimic ambient shifts.

### `JITTER_SAMPLES` *(int ≥ 0, Default: `5`)*
- **What:** Temporal shift (in samples) applied to whole cycles to emulate timing offsets.

- **Common mistakes:** Excessive augmentation that creates unrealistic data (watch validation loss).

---

## 8) Training Defaults (Schedulers & QAT)

### `MIN_EPOCHS_BEST` *(int, Default: `30`)*
- **What:** Do not consider “best model” / early stop until this epoch (and warm‑up done).
- **Why:** Stabilizes early training.

### `LR_SCHED_PLATEAU` *(bool, Default: `True`)*
- **What:** Enables LR reduction when validation stops improving.

### `LR_FACTOR` *(float, Default: `0.5`)*
- **What:** Multiply LR by this factor on plateau (e.g., 0.5 halves it).

### `LR_PATIENCE` *(int, Default: `4`)*
- **What:** Number of validations with no improvement before LR step.

### `LR_MIN` *(float, Default: `1e-6`)*
- **What:** Lower bound for LR.

### QAT observer knobs
- **`EMA_DECAY_QAT`** *(0–1, Default: `0.99`)* — How quickly running stats forget old values.  
- **`Q_DELAY_UPDATES`** *(int ≥ 0, Default: `400`)* — Wait this many updates before quantization “locks in.”  
- **`Q_FREEZE_UPDATES`** *(int ≥ 0, Default: `4000`)* — After this, observers freeze (stable export).

- **Tip:** If early training is unstable, increase `Q_DELAY_UPDATES`. For reproducible export, keep `Q_FREEZE_UPDATES` reasonably large.

---

## 9) Runtime Controls (formerly CLI)

### `DEVICE` *(str, Default: `"cuda"` if available else `"cpu"`)*
- **What:** Compute device. Override to `"cpu"` if no GPU.

### `AMP_ENABLED` *(bool, Default: `True`)*, `AMP_DTYPE` *("bf16"|"fp16", Default: "bf16")*
- **What:** Mixed‑precision training to speed up and reduce memory.
- **Guidance:** Prefer **bf16** for stability. Use **fp16** only if you need GradScaler.

### `CKPT` *(bool, Default: `False`)*, `CKPT_CHUNK` *(int, Default: `16`)*
- **What:** Activation checkpointing (saves memory by recomputing forward).
- **Model support:** Fine‑grained chunking is used only if the model exposes special kwargs; otherwise a coarse fallback is used.
- **Trade‑off:** Lower memory, more compute time.

### `TBPTT_K` *(int ≥ 0, Default: `0`)*
- **What:** Truncated BPTT every K steps if the model supports it.
- **When to use:** Very long sequences with memory limits.

### `ACCUM` *(int ≥ 1, Default: `1`)*
- **What:** Gradient accumulation steps.  
- **Effective batch:** `BATCH_SIZE × ACCUM`.  
- **Use case:** Simulate larger batches without increasing memory.

### `FUSED_ADAM` *(bool, Default: `True`)*
- **What:** Try PyTorch’s fused Adam for speed (falls back if unavailable).

### `COMPILE` *(0|1, Default: `0`)*
- **What:** Enable `torch.compile` for potential speedups (falls back if unsupported).

### `VAL_INTERVAL` *(int ≥ 1, Default: `1`)*
- **What:** Validate every K epochs.  
- **Trade‑off:** Higher → faster epochs, slower feedback.

### `VAL_MAX_BATCHES` *(int ≥ 0, Default: `0` = unlimited)*
- **What:** Cap validation batches for speed during development.

---

## 10) DataLoader Settings

### `WORKERS` *(int ≥ 0, Default: auto)*
- **What:** Number of subprocesses that load data.
- **Guidance:** Linux servers: start with `4`. Laptops/Windows: `0–2`. Watch CPU load.

### `PIN_MEMORY` *(bool, Default: `True`)*
- **What:** Faster transfers to GPU. Harmless on CPU‑only.

### `PERSIST` *(bool, Default: `True`)*
- **What:** Keep workers alive across epochs (faster steady‑state). Needs `WORKERS > 0`.

### `PREFETCH` *(int or None, Default: `4`)*
- **What:** How many batches each worker preloads. Ignored if `WORKERS=0`.

---

## 11) Temporal Mixed Precision (time‑aware scaling)

A lightweight trick that **does not change the math** of activations—only their integer scaling during training/export.

### `MP_TIME` *(0|1, Default: `0`)*
- **What:** Enable/disable the feature.

### `MP_TAU_THR` *(float, °C/s, Default: `0.08`)*
- **What:** Threshold for “fast change” detection in temperature dynamics.

### `MP_SCALE_MUL` *(float, Default: `1.5`)*
- **What:** Multiplier for gate pre‑activation scale when in fast regime.

### `MP_RSHIFT_DELTA` *(int, Default: `-1`)*
- **What:** Additional right shift in fast regime (reduces magnitude = safer integer ranges).

- **When to use:** If INT8 overflows or saturation show up around sharp transients.

---

## 12) Aliases & Legacy

### `OUTPUT_SIZE`
- Present for completeness/legacy heads (kept as `1`).

### `PLOTS_DIR`
- Alias of `PLOT_PATH` for components that expect that name.

### `EPOCHS`
- Legacy epoch count. The current loop uses `MAX_EPOCHS` and early‑stopping.

---

## 13) Interactions Cheat‑Sheet

- **Validation cadence:** Fewer validations ⇒ set `VAL_INTERVAL > 1`; combine with `PATIENCE` (e.g., `VAL_INTERVAL=2`, `PATIENCE=5`).  
- **Early stop gate:** Best‑model saving/early stop only after `max(MIN_EPOCHS_BEST, LAMBDA_WARMUP_EPOCHS)` **and** when a validation actually runs.  
- **Transient loss:** Requires `WINDOW_SIZE ≥ 2`.  
- **Effective batch:** Increase `ACCUM` if you hit memory limits.  
- **Overfitting signs:** Training loss ↓, validation loss ↑ → increase `DROPOUT`, add augmentation, or reduce `HIDDEN_SIZE`.  
- **Underfitting signs:** Both losses high → increase `HIDDEN_SIZE` or `NUM_LAYERS`, or reduce regularization.  
- **Physics too strong:** If predictions look “flattened,” reduce `LAMBDA_SS`/`LAMBDA_TR` or increase warm‑up.  
- **Unstable training:** Lower `LEARNING_RATE`; raise `Q_DELAY_UPDATES`; prefer `AMP_DTYPE="bf16"`; consider `CKPT` if memory is tight.

---

## 14) Quick Tuning Recipes

- **Faster experiments:** `VAL_MAX_BATCHES=5`, `VAL_INTERVAL=3`, `MAX_EPOCHS=200`, `PATIENCE=5`.  
- **More conservative physics:** `LAMBDA_WARMUP_EPOCHS=20`; halve both `LAMBDA_SS`, `LAMBDA_TR`.  
- **Stronger model:** Double `HIDDEN_SIZE` (e.g., 16→32); if still underfitting, set `NUM_LAYERS=2`.  
- **INT8 stability around spikes:** Enable `MP_TIME=1`, keep defaults; only adjust if needed (`MP_SCALE_MUL=2.0`, `MP_RSHIFT_DELTA=-2`).

---

## 15) Glossary (tiny)

- **Window:** Short slice of consecutive samples used as a model input.  
- **Epoch:** One full pass over the training data.  
- **Validation:** Held‑out data to monitor generalization (no training).  
- **Early stopping:** Stop training when validation stops improving.  
- **QAT (Quantization‑Aware Training):** Train the model while simulating INT8 effects to get better integer inference later.
