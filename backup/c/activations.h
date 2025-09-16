#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include <stdint.h>

int16_t sigmoid_from_pre_q8(int32_t pre_q8);
int16_t tanh_from_pre_q8   (int32_t pre_q8);


int16_t tanh_from_c_q8(int32_t c_q8);

#endif // ACTIVATIONS_H
