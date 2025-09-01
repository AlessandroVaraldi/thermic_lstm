#include "model_int8.h"
#include "activations.h"

static inline int clamp_int(int x, int lo, int hi){ return x<lo?lo:(x>hi?hi:x); }

int16_t sigmoid_from_pre_q8(int32_t pre_q8) {
    int idx = clamp_int((int)pre_q8, -127, 127) + 127;
    return (int16_t)LUT_SIG_GATE_Q8[idx]; // [0..256]
}

int16_t tanh_from_pre_q8(int32_t pre_q8) {
    int idx = clamp_int((int)pre_q8, -127, 127) + 127;
    return (int16_t)LUT_TANH_GATE_Q8[idx]; // [-256..256]
}
