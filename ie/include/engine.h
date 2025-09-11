// include/engine.h
#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------- Q formats ----------------
#define SAT8(x)  ((int8_t)(( (x) > 127) ? 127 : ((x) < -128 ? -128 : (x))))
#define SAT16(x) ((int16_t)(((x) > 32767) ? 32767 : ((x) < -32768 ? -32768 : (x))))

// mul in int64 con round-shift a destra (>=0), Q15 multiplier
static inline int32_t mul_q15_shift(int32_t acc, int16_t mult_q15, uint8_t rshift) {
    int64_t t = (int64_t)acc * (int64_t)mult_q15;         // acc * m (Q15)
    // round: add 0.5*2^r
    int64_t add = (rshift > 0) ? (int64_t)1 << (rshift-1) : 0;
    int64_t y = (t + add) >> rshift;
    // After this y is in acc units scaled by 1/2^r * Q15 => target int32
    if (y >  2147483647LL) y =  2147483647LL;
    if (y < -2147483648LL) y = -2147483648LL;
    return (int32_t)y;
}

// ---------------- Requant factors per gate ----------------
typedef struct {
    int16_t mult_q15;  // Q15 multiplier
    uint8_t rshift;    // right shift
} rq_q15_t;

typedef struct {
    rq_q15_t ih, hh;   // separate requant for ih and hh branches
} gate_rq_t;

typedef struct {
    // W_ih: [4H, in], W_hh: [4H, H] — row-major int8
    const int8_t  *W_ih;
    const int32_t *B_ih;   // int32 bias (acc domain), can be NULL
    const int8_t  *W_hh;
    const int32_t *B_hh;   // can be NULL
    int16_t H; int16_t in;
    gate_rq_t rq[4];       // 0:i 1:f 2:g 3:o
} lstm_layer_t;

typedef struct {
    const int8_t  *W;      // [1, 2H] int8
    const int32_t *B;      // int32 or NULL
    int16_t in;
} fc_head_t;

typedef struct {
    int16_t L;     // num layers
    int16_t H;     // hidden size
    lstm_layer_t *layers;
    fc_head_t     fc;
    // (opzionale) normalizzazione esterna
    float mu_y, std_y;
} qat_model_t;

// --------- Engine API ---------
void qat_bind_weights(qat_model_t *m, const uint8_t *blob_base); // usa model_offsets.h
void lstm_reset(const qat_model_t *m, int16_t *h, int16_t *c);
void lstm_step(const qat_model_t *m, const int16_t *x_q15, // x: Q0.15 features concatenati (P_norm, Tbp_norm)
               int16_t *h, int16_t *c);                    // in/out: h,c Q0.15
// attention + head → y_norm (Q0.15)
int16_t lstm_run_sequence(const qat_model_t *m,
                          const int16_t *x_q15_seq, // [W, D] in Q0.15, D deve combaciare
                          int W,
                          int16_t *scratch_h_seq,   // [W, H] Q0.15 (workspace)
                          int16_t *h_last, int16_t *c_last);
#ifdef __cplusplus
}
#endif
