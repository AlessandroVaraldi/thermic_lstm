// tanh_poly.c — tanh(c) INT8/Q8 via I-BERT i-exp (no LUT)
// Usa la tanh_int_q8 esposta da sigmoid_poly.c e indicizza c con round-shift.

#include <stdint.h>
#include "model_int8.h"
#include "activations.h"

// tanh_int_q8 è definita in sigmoid_poly.c (non static)
extern int32_t tanh_int_q8(int32_t q_x, int32_t S_q8);

// rounding shift per potenze di 2
static inline int round_shr_pow2(int x, int sh){
    if (sh <= 0) return x;
    int b = 1 << (sh - 1);
    return (x >= 0) ? ((x + b) >> sh) : ((x - b) >> sh);
}

// deduci log2(S_TANHC_Q8) a compile-time se è potenza di 2
#ifndef TANHC_SHIFT
# if ((S_TANHC_Q8 & (S_TANHC_Q8 - 1)) == 0)
#   define TANHC_SHIFT ( \
        (S_TANHC_Q8==1)?0 : (S_TANHC_Q8==2)?1 : (S_TANHC_Q8==4)?2 : \
        (S_TANHC_Q8==8)?3 : (S_TANHC_Q8==16)?4: (S_TANHC_Q8==32)?5: \
        (S_TANHC_Q8==64)?6 : (S_TANHC_Q8==128)?7 : -1)
# else
#   define TANHC_SHIFT (-1)
# endif
#endif

static inline int clamp_int(int x, int lo, int hi){ return x<lo?lo:(x>hi?hi:x); }

int16_t tanh_from_c_q8(int32_t c_q8){
    // indice q = round(c / S_TANHC_Q8) in [-127..127]
    int q;
#if (TANHC_SHIFT >= 0)
    q = round_shr_pow2((int)c_q8, TANHC_SHIFT);
#else
    q = (c_q8 >= 0) ? ((int)(c_q8 + S_TANHC_Q8/2) / (int)S_TANHC_Q8)
                    : ((int)(c_q8 - S_TANHC_Q8/2) / (int)S_TANHC_Q8);
#endif
    q = clamp_int(q, -127, 127);

    // usa tanh_int_q8 con S_TANHC_Q8 (64 tipicamente)
    int32_t y = tanh_int_q8(q, S_TANHC_Q8);
    if (y < -256) y = -256; else if (y > 256) y = 256;
    return (int16_t)y;
}
