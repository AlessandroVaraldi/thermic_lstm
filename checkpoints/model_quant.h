// Auto-generated quant header (minimal)
#pragma once

#define MODEL_INPUT_SIZE   2
#define MODEL_HIDDEN_SIZE  16
#define MODEL_NUM_LAYERS   1

// FC per-tensor scales
static const float FC_SX = 0.007866973988711834f;
static const float FC_SW = 0.002154788002371788f;

// LSTM cells per-layer scales
// Layer 0
static const float L0_IH_SX = 0.0243837907910347f;
static const float L0_IH_SW = 0.0056370398961007595f;
static const float L0_HH_SX = 0.006087375804781914f;
static const float L0_HH_SW = 0.0021808522287756205f;
static const int   L0_S_GATE_Q8  = 32;
static const int   L0_S_TANHC_Q8 = 64;

// Mixed-precision temporale (griglie tanh(c))
#define MP_TIME_ENABLED  1
static const float TANHC_GRID_BASE = 256.0f;
static const float TANHC_GRID_FINE = 768.0f;
