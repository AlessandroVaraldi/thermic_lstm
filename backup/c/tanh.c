#include "model_int8.h"
#include "activations.h"

static inline int clamp_int(int x, int lo, int hi){ return x<lo?lo:(x>hi?hi:x); }

static inline int round_div(int a, int b){
    if (a >= 0) return (a + b/2) / b;
    return (a - b/2) / b;
}

int16_t tanh_from_c_q8(int32_t c_q8) {
    int q = round_div((int)c_q8, (int)S_TANHC_Q8);
    int idx = clamp_int(q, -127, 127) + 127;
    return (int16_t)LUT_TANH_C_Q8[idx]; // [-256..256]
}
