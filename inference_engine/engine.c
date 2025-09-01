#include <math.h>
#include <string.h>
#include "engine.h"
#include "activations.h"
#include <stdio.h>

#ifndef GATE_I
#define GATE_I 1  // i
#define GATE_F 0  // f
#define GATE_G 2  // g
#define GATE_O 3  // o
#endif

#ifndef LSTM_L
#define LSTM_L 1
#endif

// ---------- helpers ----------
static inline int32_t clamp32(int32_t x, int32_t lo, int32_t hi){ return x<lo?lo:(x>hi?hi:x); }
static inline int16_t clamp16(int32_t x, int32_t lo, int32_t hi){ return (int16_t)clamp32(x, lo, hi); }

// Q8 mul: (a*b)/256 arrotondato
static inline int16_t q8_mul(int16_t a_q8, int16_t b_q8){
    int32_t t = (int32_t)a_q8 * (int32_t)b_q8;
    t = (t + (t>=0?128:-128)) >> 8;
    return clamp16(t, -32768, 32767);
}

// dot INT8·INT8 -> INT32
static inline int32_t dot_i8(const int8_t* __restrict a, const int8_t* __restrict b, int n){
    int32_t acc = 0;
    for (int i=0;i<n;i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

// quantizzazione float->int8 con scala s
static inline int8_t quant_i8(float v, float s){
    int q = (int)lrintf(v / s);
    if (q < -127) q = -127; else if (q > 127) q = 127;
    return (int8_t)q;
}

// moltiplicazione fissa con shift
static inline int32_t fx_mul_shift(int64_t x){
    const int64_t bias = (1LL<<(FX_SHIFT-1));
    return (int32_t)((x >= 0 ? x + bias : x - bias) >> FX_SHIFT);
}

// ---------- indexing dei pesi flatten ----------
#define IH_STRIDE (LSTM_4H*LSTM_F)
#define HH_STRIDE (LSTM_4H*LSTM_H)
#define IH_OFF(l,g,u)   (((l)* (LSTM_4H) + (g)*LSTM_H + (u)) * LSTM_F)
#define HH_OFF(l,g,u)   (((l)* (LSTM_4H) + (g)*LSTM_H + (u)) * LSTM_H)
#define B_OFF(l,g,u)    ((l)* (LSTM_4H) + (g)*LSTM_H + (u))

void qlstm_reset(qlstm_state_t* st){ memset(st, 0, sizeof(*st)); }

// ---------- core: un passo per layer ----------
static inline void lstm_step_layer(
    int layer,
    const int8_t* __restrict x_q, // [F] int8
    qlstm_state_t* st
){
    // quantizza una volta sola h(layer): Q8 -> int8 con la scala di input HH
    int8_t h_qi8[LSTM_H];
    for (int u=0; u<LSTM_H; ++u){
        h_qi8[u] = quant_i8((float)st->h[layer][u] / 256.0f, QX_HH_SCALES[layer]);
    }

    static int sat_pre=0, sat_c=0, tot=0;

    for (int u=0; u<LSTM_H; ++u){
        // IH contributions (dot su F)
        int32_t pre_i = fx_mul_shift((int64_t)dot_i8(x_q, &IH_W_Q8[IH_OFF(layer,GATE_I,u)], LSTM_F) * FX_M_IH[layer]);
        int32_t pre_f = fx_mul_shift((int64_t)dot_i8(x_q, &IH_W_Q8[IH_OFF(layer,GATE_F,u)], LSTM_F) * FX_M_IH[layer]);
        int32_t pre_g = fx_mul_shift((int64_t)dot_i8(x_q, &IH_W_Q8[IH_OFF(layer,GATE_G,u)], LSTM_F) * FX_M_IH[layer]);
        int32_t pre_o = fx_mul_shift((int64_t)dot_i8(x_q, &IH_W_Q8[IH_OFF(layer,GATE_O,u)], LSTM_F) * FX_M_IH[layer]);

        // HH contributions (dot su H) usando h_qi8
        pre_i += fx_mul_shift((int64_t)dot_i8(h_qi8, &HH_W_Q8[HH_OFF(layer,GATE_I,u)], LSTM_H) * FX_M_HH[layer]);
        pre_f += fx_mul_shift((int64_t)dot_i8(h_qi8, &HH_W_Q8[HH_OFF(layer,GATE_F,u)], LSTM_H) * FX_M_HH[layer]);
        pre_g += fx_mul_shift((int64_t)dot_i8(h_qi8, &HH_W_Q8[HH_OFF(layer,GATE_G,u)], LSTM_H) * FX_M_HH[layer]);
        pre_o += fx_mul_shift((int64_t)dot_i8(h_qi8, &HH_W_Q8[HH_OFF(layer,GATE_O,u)], LSTM_H) * FX_M_HH[layer]);

        // bias (Q8)
        pre_i += B_Q8[B_OFF(layer,GATE_I,u)];
        pre_f += B_Q8[B_OFF(layer,GATE_F,u)];
        pre_g += B_Q8[B_OFF(layer,GATE_G,u)];
        pre_o += B_Q8[B_OFF(layer,GATE_O,u)];

        tot += 4;
        sat_pre += (pre_i<=-127 || pre_i>=127) + (pre_f<=-127 || pre_f>=127)
                + (pre_g<=-127 || pre_g>=127) + (pre_o<=-127 || pre_o>=127);
        // dopo il clamp di c:
        if (st->c[layer][u] == (int16_t)(C_CLIP*256) || st->c[layer][u] == -(int16_t)(C_CLIP*256)) sat_c++;


        // attivazioni via LUT
        int16_t i_q8 = sigmoid_from_pre_q8(pre_i);
        int16_t f_q8 = sigmoid_from_pre_q8(pre_f);
        int16_t g_q8 = tanh_from_pre_q8   (pre_g);
        int16_t o_q8 = sigmoid_from_pre_q8(pre_o);

        // c = f*c + i*g
        int16_t fc = q8_mul(f_q8, st->c[layer][u]);
        int16_t ig = q8_mul(i_q8, g_q8);
        int32_t c_new = (int32_t)fc + (int32_t)ig;

        // clamp c a ±C_CLIP (Q8)
        int32_t c_clip_q8 = (int32_t)(C_CLIP * 256.0f);
        c_new = clamp32(c_new, -c_clip_q8, c_clip_q8);
        st->c[layer][u] = (int16_t)c_new;

        // h = o * tanh(c)
        int16_t tnh_c = tanh_from_c_q8(st->c[layer][u]);
        st->h[layer][u] = q8_mul(o_q8, tnh_c);
    }
    printf("Layer %d: sat_pre=%d, sat_c=%d, tot=%d\n", layer, sat_pre, sat_c, tot);
}

// ---------- inference su finestra ----------
float infer_window_q(const int8_t xq_win[WIN_SIZE][LSTM_F], qlstm_state_t* st){
    static float Y_seq[WIN_SIZE][LSTM_H];

    for (int t=0; t<WIN_SIZE; ++t){
        const int8_t* x_in = xq_win[t];

        // layer 0
        lstm_step_layer(0, x_in, st);

        // layers > 0: re-quantizza h(l-1) -> int8 con QX_IH_SCALES[l]
        for (int l=1; l<LSTM_L; ++l){
            int8_t xq_buf[LSTM_H]; // F==H per layer interni
            for (int u=0; u<LSTM_H; ++u){
                xq_buf[u] = quant_i8((float)st->h[l-1][u] / 256.0f, QX_IH_SCALES[l]);
            }
            lstm_step_layer(l, xq_buf, st);
        }

        for (int u=0; u<LSTM_H; ++u)
            Y_seq[t][u] = (float)st->h[LSTM_L-1][u] / 256.0f;
    }
    
    // attenzione semplice: dot(H_last, Y_t) -> softmax -> contesto
    float H_last[LSTM_H];
    for (int u=0; u<LSTM_H; ++u) H_last[u] = (float)st->h[LSTM_L-1][u] / 256.0f;

    #if 1
    float scores[WIN_SIZE], maxs = -1e30f;
    for (int t=0; t<WIN_SIZE; ++t){
        float s = 0.f;
        for (int u=0; u<LSTM_H; ++u) s += Y_seq[t][u] * H_last[u];
        scores[t] = s; if (s > maxs) maxs = s;
    }
    float sum = 0.f;
    for (int t=0; t<WIN_SIZE; ++t){ scores[t] = expf(scores[t] - maxs); sum += scores[t]; }
    for (int t=0; t<WIN_SIZE; ++t)  scores[t] /= (sum + 1e-12f);

    float ctx[LSTM_H];
    for (int u=0; u<LSTM_H; ++u){
        float acc = 0.f;
        for (int t=0; t<WIN_SIZE; ++t) acc += scores[t] * Y_seq[t][u];
        ctx[u] = acc;
    }
    #else
    float ctx[LSTM_H];
    for (int u=0; u<LSTM_H; ++u) ctx[u] = 0.f;   // niente contesto
    #endif
    
    // testa FC (float) su [H_last | ctx]
    float y = 0.f;
    for (int u=0; u<LSTM_H; ++u) y += FC_W[u] * H_last[u];
    for (int u=0; u<LSTM_H; ++u) y += FC_W[LSTM_H + u] * ctx[u];
    y += FC_B[0];

    // denormalizza
    y = y * STD_Y + MU_Y;
    return y;
}

float infer_window(const float x_win[WIN_SIZE][LSTM_F], qlstm_state_t* st){
    static int8_t xq_win[WIN_SIZE][LSTM_F];

    for (int t=0; t<WIN_SIZE; ++t){
        for (int f=0; f<LSTM_F; ++f){
            float xn = (x_win[t][f] - MU_X[f]) / (STD_X[f] + 1e-6f);
            xq_win[t][f] = quant_i8(xn, QX_IH_SCALES[0]); // scala del primo layer
        }
    }
    return infer_window_q(xq_win, st);
}
