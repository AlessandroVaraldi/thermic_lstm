// src/acts.h
#pragma once
#include <stdint.h>

// ======= Config =======
#ifndef USE_LUT_ACTS
#define USE_LUT_ACTS 0   // 0 = PWL ultrarapide; 1 = usa tabelle (da fornire)
#endif

// Σ: int8 pre → uint8 Q0.8
static inline uint8_t sigmoid_q8(int8_t x)
{
#if USE_LUT_ACTS
    extern const uint8_t SIGM_LUT_256[256];
    return SIGM_LUT_256[(uint8_t)x]; // x interpretato come indice 0..255 con offset 128
#else
    // Fast "x/(1+|x|)" mapped to [0,1], tuned for Q8 input
    // x in [-128..127] → map to approx sigma
    int16_t xi = (int16_t)x;
    int16_t ax = (xi < 0) ? -xi : xi;           // |x|
    int16_t denom = (int16_t)128 + ax;          // 128 ~ slope scale
    int32_t num = ((int32_t)xi << 7);           // x * 128
    int32_t frac = num / denom;                 // ~[-128..127]
    int16_t yq = (int16_t)(frac + 128);         // shift to [0..255]
    if (yq < 0) yq = 0; if (yq > 255) yq = 255;
    return (uint8_t)yq;
#endif
}

// tanh: int8 pre → int8 Q0.8
static inline int8_t tanh_q8(int8_t x)
{
#if USE_LUT_ACTS
    extern const int8_t TANH_LUT_256[256];
    return TANH_LUT_256[(uint8_t)x];
#else
    // Odd function, piecewise linear: slope reduces for large |x|
    int16_t xi = (int16_t)x;
    int16_t ax = (xi < 0) ? -xi : xi;
    // 3-piece slope: hi|mid|low
    int16_t slope;
    if (ax < 32)       slope = 120;   // near 1.0 in Q7
    else if (ax < 64)  slope = 80;    // 0.625
    else               slope = 48;    // 0.375
    int32_t y = (xi * slope) >> 7;    // Q8 * Q7 -> Q8
    if (y > 127) y = 127; if (y < -128) y = -128;
    return (int8_t)y;
#endif
}

// tanh per c(t): int16 Q0.15 → int16 Q0.15
static inline int16_t tanh_q15(int16_t x)
{
#if USE_LUT_ACTS
    // 512-entry LUT sui 9 bit alti di |x| (odd)
    extern const int16_t TANH15_LUT_512[512];
    int16_t s = x >= 0 ? 1 : -1;
    uint16_t ax = (uint16_t)(x >= 0 ? x : -x);
    uint16_t idx = ax >> 7; // usa i 9 bit alti (>>7)
    int16_t y = TANH15_LUT_512[idx];
    return (s > 0) ? y : (int16_t)(-y);
#else
    // PWL 4-piece su Q0.15, clamp oltre ~2.0
    int32_t xi = x;
    int32_t ax = (xi >= 0) ? xi : -xi;
    if (ax >= (int32_t)(2<<15))  // |x| >= 2 → tanh≈±0.964, clamp vicino 1
        return (xi >= 0) ? 31129 : -31129; // ~0.95 in Q15
    // segments boundaries ~ 0.5,1.0,1.5
    const int32_t b0 = (int32_t)(0.5 * (1<<15));
    const int32_t b1 = (int32_t)(1.0 * (1<<15));
    const int32_t b2 = (int32_t)(1.5 * (1<<15));
    int32_t slope; // Q15
    if (ax < b0)      slope = (int32_t)(0.98 * (1<<15));
    else if (ax < b1) slope = (int32_t)(0.85 * (1<<15));
    else if (ax < b2) slope = (int32_t)(0.65 * (1<<15));
    else              slope = (int32_t)(0.45 * (1<<15));
    int32_t y = ( (xi * slope) >> 15 );
    if (y > 32767) y = 32767; if (y < -32768) y = -32768;
    return (int16_t)y;
#endif
}
