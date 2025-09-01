#ifndef ENGINE_H
#define ENGINE_H
#include <stdint.h>
#include "model_int8.h"  

#ifdef __cplusplus
extern "C" {
#endif

// Stato per L layers (Q8: 1.0 -> 256)
typedef struct {
    int16_t h[LSTM_L][LSTM_H];
    int16_t c[LSTM_L][LSTM_H];
} qlstm_state_t;

// Azzera lo stato
void qlstm_reset(qlstm_state_t* st);

// Inference su una finestra float [WIN_SIZE x LSTM_F] (normalizza, quantizza se disponibili le scale) -> output float denormalizzato
float infer_window(const float x_win[WIN_SIZE][LSTM_F], qlstm_state_t* st);

// Variante: input già quantizzato int8 [WIN_SIZE x LSTM_F] (usa direttamente dot INT8)
float infer_window_q(const int8_t xq_win[WIN_SIZE][LSTM_F], qlstm_state_t* st);

#ifdef __cplusplus
}
#endif
#endif // ENGINE_H
