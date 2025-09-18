// acts.h
#pragma once
#include <stdint.h>

// LUT-based gate activations (pre ∈ [-128..127])
static inline uint8_t sigmoid_uq8_lut(int8_t pre_s8, const uint8_t *lut256){
  return lut256[(uint8_t)(pre_s8 + 128)];
}
static inline int8_t tanh_s8_lut(int8_t pre_s8, const int8_t *lut256){
  return lut256[(uint8_t)(pre_s8 + 128)];
}

// delta |Δx_norm| in Q4.12 from INT8 normalized inputs (per step)
static inline uint16_t delta_q12_from_x8(const int8_t *x_prev, const int8_t *x_cur, int D){
  int maxd = 0;
  for(int k=0;k<D;k++){
    int d = (int)x_cur[k] - (int)x_prev[k];
    if(d<0) d = -d;
    if(d > maxd) maxd = d;
  }
  // map |Δs8| (0..255) -> Q4.12 of normalized delta ≈ (Δ/127)
  // delta_q12 = round( Δ * 4096 / 127 )
  return (uint16_t)((maxd * 4096 + 63) / 127);
}

// tanh(c) with optional MP-time (quarter-step interpolation)
static inline int8_t tanh_c_s8_mp(
  int16_t c_q12, const int8_t *lut65536,
  int mp_enabled, uint16_t tau_thr_q12, uint16_t delta_q12
){
  uint16_t idx = (uint16_t)(c_q12 + (int32_t)32768);
  if(!mp_enabled || delta_q12 < tau_thr_q12){
    return lut65536[idx];
  }
  // 4x sub-stepping using low 2 bits as fraction
  uint32_t idx4 = ((uint32_t)idx) << 2;
  uint16_t base = (uint16_t)(idx4 >> 2);
  uint8_t  frac = (uint8_t)(idx4 & 0x3);
  int16_t v0 = lut65536[base];
  int16_t v1 = lut65536[(uint16_t)(base + (base==65535?0:1))];
  int16_t num = (int16_t)(( (int32_t)v0*(4-frac) + (int32_t)v1*frac ) >> 2);
  return (int8_t)num;
}

// Softmax-like weights (Q1.15) using base-2 exponent (integer-only)
static inline void softmax_pow2_q15_from_scores(
  const int32_t *scores, int T, int16_t *w_q15
){
  // find max
  int32_t mx = scores[0];
  for(int t=1;t<T;t++) if(scores[t]>mx) mx = scores[t];
  // find max delta
  int32_t md = 0;
  for(int t=0;t<T;t++){
    int32_t d = mx - scores[t];
    if(d > md) md = d;
  }
  // choose shift s.t. (md >> s) <= 14
  int s = 0;
  while( ((md >> s) > 14) && (s < 24) ) s++;
  // exp2 weight = 1 << (14 - k), k=(delta>>s) clamped to [0..14]
  int32_t sum = 0;
  for(int t=0;t<T;t++){
    int32_t d = mx - scores[t];
    int k = (int)(d >> s);
    if(k > 14) k = 14;
    int32_t e = 1 << (14 - k);  // Q0 integer
    w_q15[t] = (int16_t)e;      // temp store integer weight
    sum += e;
  }
  // normalize to Q1.15 via integer division
  if(sum <= 0){ for(int t=0;t<T;t++) w_q15[t] = (int16_t)( (1<<15)/T ); return; }
  for(int t=0;t<T;t++){
    int32_t e = (int32_t)w_q15[t];                // integer
    int32_t q = ( (e << 15) + (sum>>1) ) / sum;   // Q1.15
    if(q > 32767) q = 32767;
    w_q15[t] = (int16_t)q;
  }
}
