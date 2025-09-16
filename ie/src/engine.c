// src/engine.c
#include <stdint.h>
#include <string.h>
#include "engine.h"
#include "acts.h"
#include "include/model_offsets.h"

// ---- fixed-size scratch ----
#define MAX_IN  (MODEL_INPUT)
#define MAX_H   (MODEL_HIDDEN)

// ---- tiny helpers ----
static inline int8_t  sat8 (int32_t v){ if(v>127) return 127; if(v<-128) return -128; return (int8_t)v; }
static inline int16_t sat16(int32_t v){ if(v>32767) return 32767; if(v<-32768) return -32768; return (int16_t)v; }
static inline int8_t  q15_to_q8(int16_t v){ return sat8(((int32_t)v) >> 7); }
static inline int32_t dot_s8_s8(const int8_t *w, const int8_t *x, int n){
    int32_t a=0; for(int i=0;i<n;++i) a += (int32_t)w[i]*(int32_t)x[i]; return a;
}
static inline int8_t  rq_q8 (int32_t acc, int16_t m, int8_t s){ return sat8(mul_q15_shift(acc, m, s)); }
static inline int16_t rq_q15(int32_t acc, int16_t m, int8_t s){ return sat16(mul_q15_shift(acc, m, s)); }

// Optional ABI-compat for acts.h variants.
static inline uint8_t sig_q8_scaled(int8_t x, int S){
#ifdef ACTS_SIGMOID_Q8_TAKES_SCALE
    return sigmoid_q8(x, S);
#else
    (void)S; return sigmoid_q8(x);
#endif
}
static inline int8_t tanh_q8_scaled(int8_t x, int S){
#ifdef ACTS_TANH_Q8_TAKES_SCALE
    return tanh_q8(x, S);
#else
    (void)S; return tanh_q8(x);
#endif
}

// ---- normalization: x_norm = a*x + b (Q15) ----
static inline void norm_inputs_q15(const int16_t in[MODEL_INPUT], int16_t out[MODEL_INPUT]){
#if MODEL_INPUT != 2
# error "This engine assumes MODEL_INPUT == 2"
#endif
    int32_t t0 = ((int32_t)NORM_AX0_Q15 * (int32_t)in[0]) >> 15; t0 += (int32_t)NORM_BX0_Q15;
    int32_t t1 = ((int32_t)NORM_AX1_Q15 * (int32_t)in[1]) >> 15; t1 += (int32_t)NORM_BX1_Q15;
    out[0] = sat16(t0); out[1] = sat16(t1);
}

// ================= Bind =================
void qat_bind_weights(qat_model_t *m, const uint8_t *blob)
{
    m->L = MODEL_LAYERS;
    m->H = MODEL_HIDDEN;

    for (int l=0; l<m->L; ++l) {
        lstm_layer_t *Lr = &m->layers[l];
        switch (l) {
        #define BIND_L(i) do{ \
            Lr->W_ih = (const int8_t *)(blob + L##i##_IH_W_OFF); \
            Lr->B_ih = (L##i##_IH_B_NBYTES ? (const int32_t*)(blob + L##i##_IH_B_OFF) : NULL); \
            Lr->W_hh = (const int8_t *)(blob + L##i##_HH_W_OFF); \
            Lr->B_hh = (L##i##_HH_B_NBYTES ? (const int32_t*)(blob + L##i##_HH_B_OFF) : NULL); \
            Lr->in   = (int16_t)L##i##_IH_IN; \
            Lr->H    = (int16_t)(L##i##_IH_OUT/4); \
            Lr->S_gate_q8  = (int16_t)L##i##_S_GATE_Q8; \
            Lr->S_tanhc_q8 = (int16_t)L##i##_S_TANHC_Q8; \
            Lr->rq[0].ih.mult_q15 = L##i##_IH_RQ_M_i; Lr->rq[0].ih.rshift = L##i##_IH_RQ_S_i; \
            Lr->rq[0].hh.mult_q15 = L##i##_HH_RQ_M_i; Lr->rq[0].hh.rshift = L##i##_HH_RQ_S_i; \
            Lr->rq[1].ih.mult_q15 = L##i##_IH_RQ_M_f; Lr->rq[1].ih.rshift = L##i##_IH_RQ_S_f; \
            Lr->rq[1].hh.mult_q15 = L##i##_HH_RQ_M_f; Lr->rq[1].hh.rshift = L##i##_HH_RQ_S_f; \
            Lr->rq[2].ih.mult_q15 = L##i##_IH_RQ_M_g; Lr->rq[2].ih.rshift = L##i##_IH_RQ_S_g; \
            Lr->rq[2].hh.mult_q15 = L##i##_HH_RQ_M_g; Lr->rq[2].hh.rshift = L##i##_HH_RQ_S_g; \
            Lr->rq[3].ih.mult_q15 = L##i##_IH_RQ_M_o; Lr->rq[3].ih.rshift = L##i##_IH_RQ_S_o; \
            Lr->rq[3].hh.mult_q15 = L##i##_HH_RQ_M_o; Lr->rq[3].hh.rshift = L##i##_HH_RQ_S_o; \
        }while(0)
        case 0: BIND_L(0); break;
        #if MODEL_LAYERS > 1
        case 1: BIND_L(1); break;
        #endif
        #if MODEL_LAYERS > 2
        case 2: BIND_L(2); break;
        #endif
        default: break;
        #undef BIND_L
        }
    }
    m->fc.W   = (const int8_t *)(blob + FC_W_OFF);
    m->fc.B   = (FC_B_NBYTES ? (const int32_t*)(blob + FC_B_OFF) : NULL);
    m->fc.in  = (int16_t)FC_IN;
    m->fc.rqm = (int16_t)FC_RQ_M;
    m->fc.rqs = (int8_t )FC_RQ_S;
}

// ================= State =================
void lstm_reset(const qat_model_t *m, int16_t *h, int16_t *c)
{
    (void)m;
    for (int i=0;i<QAT_STATE_LEN;++i){ h[i]=0; c[i]=0; }
}

// ================= One step (normalized input) =================
void lstm_step_normed(const qat_model_t *m,
                      const int16_t *x_norm_q15,
                      int16_t *h, int16_t *c,
                      int16_t *y_norm_q15)
{
    const int H = m->H;
    int8_t  x_q8[MAX_IN];
    int8_t  hprev_q8[MAX_H];
    int8_t  z_fc_q8[2*MAX_H];

    for (int l=0; l<m->L; ++l){
        const lstm_layer_t *Lr = &m->layers[l];
        const int IN = Lr->in;

        // input to Q8
        if (l==0){
            for (int d=0; d<IN; ++d) x_q8[d] = q15_to_q8(x_norm_q15[d]);
        } else {
            for (int d=0; d<IN; ++d) x_q8[d] = q15_to_q8(h[(l-1)*H + d]);
        }
        // h_{t-1} to Q8
        for (int d=0; d<H; ++d) hprev_q8[d] = q15_to_q8(h[l*H + d]);

        // unit loop
        for (int u=0; u<H; ++u){
            const int ri = 0*H + u, rf = 1*H + u, rg = 2*H + u, ro = 3*H + u;

            // IH path
            int32_t ai = dot_s8_s8(Lr->W_ih + ri*IN, x_q8, IN);
            int32_t af = dot_s8_s8(Lr->W_ih + rf*IN, x_q8, IN);
            int32_t ag = dot_s8_s8(Lr->W_ih + rg*IN, x_q8, IN);
            int32_t ao = dot_s8_s8(Lr->W_ih + ro*IN, x_q8, IN);
            if (Lr->B_ih){ ai += Lr->B_ih[ri]; af += Lr->B_ih[rf]; ag += Lr->B_ih[rg]; ao += Lr->B_ih[ro]; }
            int8_t pi = rq_q8(ai, Lr->rq[0].ih.mult_q15, Lr->rq[0].ih.rshift);
            int8_t pf = rq_q8(af, Lr->rq[1].ih.mult_q15, Lr->rq[1].ih.rshift);
            int8_t pg = rq_q8(ag, Lr->rq[2].ih.mult_q15, Lr->rq[2].ih.rshift);
            int8_t po = rq_q8(ao, Lr->rq[3].ih.mult_q15, Lr->rq[3].ih.rshift);

            // HH path
            int32_t bi = dot_s8_s8(Lr->W_hh + ri*H, hprev_q8, H);
            int32_t bf = dot_s8_s8(Lr->W_hh + rf*H, hprev_q8, H);
            int32_t bg = dot_s8_s8(Lr->W_hh + rg*H, hprev_q8, H);
            int32_t bo = dot_s8_s8(Lr->W_hh + ro*H, hprev_q8, H);
            if (Lr->B_hh){ bi += Lr->B_hh[ri]; bf += Lr->B_hh[rf]; bg += Lr->B_hh[rg]; bo += Lr->B_hh[ro]; }
            int8_t qi = rq_q8(bi, Lr->rq[0].hh.mult_q15, Lr->rq[0].hh.rshift);
            int8_t qf = rq_q8(bf, Lr->rq[1].hh.mult_q15, Lr->rq[1].hh.rshift);
            int8_t qg = rq_q8(bg, Lr->rq[2].hh.mult_q15, Lr->rq[2].hh.rshift);
            int8_t qo = rq_q8(bo, Lr->rq[3].hh.mult_q15, Lr->rq[3].hh.rshift);

            // sum and act
            int8_t pre_i = sat8((int16_t)pi + (int16_t)qi);
            int8_t pre_f = sat8((int16_t)pf + (int16_t)qf);
            int8_t pre_g = sat8((int16_t)pg + (int16_t)qg);
            int8_t pre_o = sat8((int16_t)po + (int16_t)qo);

            uint8_t i_q = sig_q8_scaled(pre_i, Lr->S_gate_q8);
            uint8_t f_q = sig_q8_scaled(pre_f, Lr->S_gate_q8);
            int8_t  g_q = tanh_q8_scaled(pre_g, Lr->S_tanhc_q8);
            uint8_t o_q = sig_q8_scaled(pre_o, Lr->S_gate_q8);

            // c,h update (Q15)
            const int idx = l*H + u;
            int32_t c32 = ((int32_t)f_q * (int32_t)c[idx]) >> 8;
            c32 += ((int32_t)i_q * (int32_t)((int16_t)g_q << 7)) >> 8;
            c[idx] = sat16(c32);
            int16_t th = tanh_q15(c[idx]);
            int32_t h32 = ((int32_t)o_q * (int32_t)th) >> 8;
            h[idx] = sat16(h32);
        }
    }

    // FC over [h_L, tanh(c_L)] → y_norm Q15
    for (int u=0; u<H; ++u){
        z_fc_q8[u]   = q15_to_q8(h[(m->L-1)*H + u]);
        int16_t th   = tanh_q15(c[(m->L-1)*H + u]);
        z_fc_q8[H+u] = q15_to_q8(th);
    }
    int32_t acc = dot_s8_s8(m->fc.W, z_fc_q8, m->fc.in);
    if (m->fc.B) acc += m->fc.B[0];
    *y_norm_q15 = rq_q15(acc, m->fc.rqm, m->fc.rqs);
}

// ================= One step (raw input) =================
void lstm_step_raw(const qat_model_t *m,
                   const int16_t *x_raw_q15,
                   int16_t *h, int16_t *c,
                   int16_t *y_norm_q15)
{
    int16_t x_norm_q15[MODEL_INPUT];
    norm_inputs_q15(x_raw_q15, x_norm_q15);
    lstm_step_normed(m, x_norm_q15, h, c, y_norm_q15);
}
