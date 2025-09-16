// sigmoid_poly.c — INT8/Q8 activations via I-BERT i-exp (no LUT)
// - Sigmoid gate: sigmoid_from_pre_q8()
// - Tanh gate:    tanh_from_pre_q8()
// - Also exposes: sigmoid_int_q8(), tanh_int_q8() for other translation units

#include <stdint.h>
#include "model_int8.h"
#include "activations.h"

// ----------------- Q8 constants -----------------
#define ONE_Q8 256
#define LN2_Q8 177

// Coefficenti default (usati per i gate)
#ifndef A_Q8
#define A_Q8  92
#endif
#ifndef B_Q8
#define B_Q8  346
#endif
#ifndef C_Q8
#define C_Q8  88
#endif

// Coefficenti dedicati per tanh(c) (S_TANHC_Q8=64) — trovati via search
#ifndef A_TANHC_Q8
#define A_TANHC_Q8 100
#endif
#ifndef B_TANHC_Q8
#define B_TANHC_Q8 331
#endif
#ifndef C_TANHC_Q8
#define C_TANHC_Q8  92
#endif

// ----------------- helpers con wrapping 32 bit -----------------
static inline int clamp_int(int x, int lo, int hi){ return x<lo?lo:(x>hi?hi:x); }
static inline uint32_t u32(uint64_t x){ return (uint32_t)(x & 0xFFFFFFFFu); }
static inline int32_t  s32(uint64_t x){ return (int32_t)u32(x); }

static inline int32_t add32(int32_t a, int32_t b){
    uint32_t ua=(uint32_t)a, ub=(uint32_t)b; return (int32_t)((ua+ub)&0xFFFFFFFFu);
}
static inline int32_t sub32(int32_t a, int32_t b){
    uint32_t ua=(uint32_t)a, ub=(uint32_t)b; return (int32_t)((ua-ub)&0xFFFFFFFFu);
}
static inline int32_t mul32(int32_t a, int32_t b){
    uint64_t p=(uint64_t)(uint32_t)a*(uint64_t)(uint32_t)b; return (int32_t)(p & 0xFFFFFFFFu);
}
static inline int32_t sar32(int32_t x, int n){
    if (n<=0) return x;
    if (x>=0) return (int32_t)((uint32_t)x >> n);
    int32_t ax=-x, adj=(1<<n)-1, q=(int32_t)((uint32_t)(ax+adj)>>n); return -q;
}

// Q8 multiply/divide (round to nearest; ties -> +inf)
static inline int32_t q8_mul32(int32_t a_q8, int32_t b_q8){
    int32_t prod = mul32(a_q8, b_q8);               // Q16 wrap
    int32_t add  = add32(prod, 128);                // +0.5 ulp
    return sar32(add, 8);                            // -> Q8
}
static inline int32_t q8_div32(int32_t n_q8, int32_t d_q8){
    if (d_q8==0) return (n_q8>=0)?0x7FFFFFFF:0x80000000;
    int64_t num=((int64_t)n_q8)<<8;
    if ((num ^ d_q8) >= 0) num += (d_q8/2); else num -= (d_q8/2);
    int64_t q = num / d_q8; return (int32_t)(q & 0xFFFFFFFF);
}

// ----------------- i-exp (I-BERT) -----------------
// versione parametrica: consente coeff diversi per tanh(c)
static inline int32_t int_exp_ibert_q8_cfg(int32_t q_y, int32_t S_q8,
                                           int32_t Aq8, int32_t Bq8, int32_t Cq8){
    if (S_q8<=0) S_q8=1;
    int32_t q_ln2 = (int32_t)((uint32_t)LN2_Q8 / (uint32_t)S_q8);
    if (q_ln2<=0) q_ln2=1;
    if (q_y>0) q_y=0;

    // y = -z*ln2 + p  (in unità "q"); p in [0, ln2)
    int32_t z  = (int32_t)((uint32_t)(-q_y) / (uint32_t)q_ln2);
    int32_t qp = add32(q_y, mul32(z, q_ln2));       // residuo (<=0)
    int32_t p_q8 = mul32(qp, S_q8);                 // Q8

    // L(p) = a*(p+b)^2 + c  -> Q8
    int32_t t_q8   = add32(p_q8, Bq8);
    int32_t t2_q16 = mul32(t_q8, t_q8);
    int32_t a_t2   = mul32(Aq8, t2_q16);            // Q24 wrap
    int32_t a_t2_r = add32(a_t2, (1<<15));          // +0.5 ulp @Q16
    int32_t a_term = sar32(a_t2_r, 16);             // -> Q8
    int32_t L_q8   = add32(a_term, Cq8);

    if (z>=31) return 0;
    int32_t exp_q8 = (int32_t)((uint32_t)L_q8 >> z);
    return (exp_q8<0)?0:exp_q8;
}

// wrapper per i gate (coeff "di default")
static inline int32_t int_exp_ibert_q8(int32_t q_y, int32_t S_q8){
    return int_exp_ibert_q8_cfg(q_y, S_q8, A_Q8, B_Q8, C_Q8);
}

// ----------------- activations (Q8) -----------------
int32_t sigmoid_int_q8(int32_t q_x, int32_t S_q8){
    if (q_x>=0){
        int32_t e_q8 = int_exp_ibert_q8(-q_x, S_q8);
        int32_t denom = add32(ONE_Q8, e_q8);
        return q8_div32(ONE_Q8, denom);
    } else {
        int32_t e_q8 = int_exp_ibert_q8(q_x, S_q8);
        int32_t denom = add32(ONE_Q8, e_q8);
        return q8_div32(e_q8, denom);
    }
}

// tanh in Q8 con i-exp (stessa logica), ma:
// - per S_q8 > 32 (tanh(c) con S=64) usa rescaling interno S_e=16 + coeff dedicati
// - per S_q8 <= 32 (gate) usa coeff di default e S_e = S_q8
int32_t tanh_int_q8(int32_t q_x, int32_t S_q8){
    int32_t sgn   = (q_x >= 0) ? 1 : -1;
    int32_t q_abs = (q_x >= 0) ? q_x : -q_x;
    if (q_abs > 127) q_abs = 127;

    int32_t use_A = A_Q8, use_B = B_Q8, use_C = C_Q8;
    int32_t S_e   = S_q8;

    if (S_q8 > 32){
        // path tanh(c): p troppo grossolano con S=64 -> usa scala più fine
        S_e   = 16;
        use_A = A_TANHC_Q8;
        use_B = B_TANHC_Q8;
        use_C = C_TANHC_Q8;
    }

    // q' = round( 2*|x| * S_q8 / S_e )
    int64_t num = (int64_t)2 * (int64_t)q_abs * (int64_t)S_q8;
    int32_t qprime = (int32_t)((num + (S_e/2)) / S_e);

    // e = exp(-2|x|) alla scala S_e con i coeff selezionati
    int32_t e_q8 = int_exp_ibert_q8_cfg(-qprime, S_e, use_A, use_B, use_C);

    // tanh = (1 - e)/(1 + e)
    int32_t nume = sub32(ONE_Q8, e_q8);
    int32_t deno = add32(ONE_Q8, e_q8);
    int32_t y    = q8_div32(nume, deno); // [0..256]
    return (sgn > 0) ? y : -y;           // [-256..256]
}

// ----------------- API chiamate dal motore -----------------
int16_t sigmoid_from_pre_q8(int32_t pre_q8){
    int32_t q = clamp_int(pre_q8, -127, 127);
    int32_t y = sigmoid_int_q8(q, S_GATE_Q8);
    if (y<0) y=0; 
    if (y>256) y=256;
    return (int16_t)y;
}

int16_t tanh_from_pre_q8(int32_t pre_q8){
    int32_t q = clamp_int(pre_q8, -127, 127);
    int32_t y = tanh_int_q8(q, S_GATE_Q8);      // per i gate S<=32: coeff default
    if (y<-256) y=-256; 
    if (y>256) y=256;
    return (int16_t)y;
}
