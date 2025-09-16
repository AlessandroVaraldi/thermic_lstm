/* ─────────────────────── lstm_model.h (MCU‑optimized) ─────────────────────── */
#pragma once
#include <stddef.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/*------- Compile-time configuration -------*/
#define INPUT_SIZE    2
#define HIDDEN_SIZE   4
#define NUM_LAYERS    1
#define BATCH_SIZE    2
#define WINDOW_SIZE   3
#define GATE_COUNT   (4 * HIDDEN_SIZE)

/* Work buffer policy: 0 = on stack (default), 1 = static (zero‑init in .bss) */
#ifndef USE_STATIC_WORKBUF
#define USE_STATIC_WORKBUF 0
#endif

/* Fast-math policy: 1 = inline approx (riduce code‑size / call overhead) */
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 1
#endif

/* Hint per optimizer (C99) */
#ifndef RESTRICT
#define RESTRICT restrict
#endif

/*------- Math utils -------*/
#if USE_FAST_MATH
/* Polinomiale stabile/cheap per tanh, err ~1e‑3 su |x|<=5 */
static inline float fast_tanhf(float x){
    const float x2 = x * x;
    return x * (27.f + x2) / (27.f + 9.f * x2);
}
static inline float fast_expf(float x){
    /* Approccio piecewise: exp(x) ≈ 2^(x/ln2) con correzione quadratica */
    const float inv_ln2 = 1.4426950408889634f;   /* 1/ln(2) */
    float y = x * inv_ln2;
    int   yi = (int) (y > 0 ? y + 0.5f : y - 0.5f);
    float r = (y - yi) * 0.6931471805599453f;    /* (y-yi)*ln2 */
    /* 2^yi via ldexpf + exp(r) ~ 1 + r + r^2/2 */
    float er = 1.f + r + 0.5f * r * r;
    return ldexpf(er, yi);
}
static inline float fast_sigmoidf(float x){ return 0.5f * (1.f + fast_tanhf(0.5f * x)); }
#define TANHF   fast_tanhf
#define EXPPF   fast_expf
#define SIGMOID fast_sigmoidf
#else
static inline float SIGMOID(float x){ return 1.f / (1.f + expf(-x)); }
#define TANHF   tanhf
#define EXPPF   expf
#endif

/*------- Weight container (tienili const → ROM/Flash) -------*/
typedef struct {
    /* LSTM – un blocco per layer (row-major: gate-major) */
    const float W_ih[NUM_LAYERS][GATE_COUNT * INPUT_SIZE];   /* (4H, F)  */
    const float W_hh[NUM_LAYERS][GATE_COUNT * HIDDEN_SIZE];  /* (4H, H)  */
    const float b_ih[NUM_LAYERS][GATE_COUNT];
    const float b_hh[NUM_LAYERS][GATE_COUNT];
    /* FC head */
    const float fc_weight[2 * HIDDEN_SIZE];                  /* (1, 2H)  */
    const float fc_bias;                                     /* scalar   */
} LSTMModelWeights;

/*------- Stato esterno: rende la funzione re-entrante -------*/
typedef struct {
    float h[NUM_LAYERS][BATCH_SIZE][HIDDEN_SIZE];
    float c[NUM_LAYERS][BATCH_SIZE][HIDDEN_SIZE];
} LSTMState;

static inline void lstm_reset(LSTMState * RESTRICT s){
    memset(s, 0, sizeof(*s));
}

/*------- Forward: sequence (B,W,F) → B-sized vector -------*/
static void lstm_forward(const float * RESTRICT seq,   /* (B,W,F) row-major        */
                         LSTMState * RESTRICT st,
                         const LSTMModelWeights * RESTRICT w,
                         float * RESTRICT out)         /* (B,)                     */
{
    /* ---- Output del last-layer per attenzione (W,B,H) ---- */
#if USE_STATIC_WORKBUF
    static float outputs[WINDOW_SIZE][BATCH_SIZE][HIDDEN_SIZE];
#else
    float outputs[WINDOW_SIZE][BATCH_SIZE][HIDDEN_SIZE];
#endif

    /* ---- LSTM pass su tutti i timestep ---- */
    for (int t = 0; t < WINDOW_SIZE; ++t) {
        const float * RESTRICT x_t0 = seq + t * INPUT_SIZE; /* inizio riga batch=0 */

        for (int l = 0; l < NUM_LAYERS; ++l) {
            const float * RESTRICT W_ih = w->W_ih[l];
            const float * RESTRICT W_hh = w->W_hh[l];

            /* Bias combinati (invarianti su t,b) */
            float b_sum[GATE_COUNT];
            for (int g = 0; g < GATE_COUNT; ++g)
                b_sum[g] = w->b_ih[l][g] + w->b_hh[l][g];

            for (int b = 0; b < BATCH_SIZE; ++b) {
                const float * RESTRICT x_b_t = x_t0 + b * (WINDOW_SIZE * INPUT_SIZE);
                const float * RESTRICT inp   = (l == 0) ? x_b_t : st->h[l-1][b];

                /* Gate pre-activations: y = W_ih*x + W_hh*h_prev + b */
                float gates[GATE_COUNT];
                for (int g = 0; g < GATE_COUNT; ++g) {
                    const float * RESTRICT wih = W_ih + g * INPUT_SIZE;
                    const float * RESTRICT whh = W_hh + g * HIDDEN_SIZE;

                    /* accumulatore col bias pre-sommato */
                    float acc = b_sum[g];

                    /* W_ih * x_t */
                    for (int i = 0; i < INPUT_SIZE; ++i) acc += wih[i] * inp[i];
                    /* W_hh * h_{t-1} */
                    const float * RESTRICT hprev = st->h[l][b];
                    for (int j = 0; j < HIDDEN_SIZE; ++j) acc += whh[j] * hprev[j];

                    gates[g] = acc;
                }

                /* Split + attivazioni (i,f,g,o) */
                float * RESTRICT i_ptr = gates;
                float * RESTRICT f_ptr = gates + HIDDEN_SIZE;
                float * RESTRICT g_ptr = gates + 2 * HIDDEN_SIZE;
                float * RESTRICT o_ptr = gates + 3 * HIDDEN_SIZE;

                float * RESTRICT cvec = st->c[l][b];
                float * RESTRICT hvec = st->h[l][b];

                for (int j = 0; j < HIDDEN_SIZE; ++j) {
                    const float i_t = SIGMOID(i_ptr[j]);
                    const float f_t = SIGMOID(f_ptr[j]);
                    const float g_t = TANHF  (g_ptr[j]);
                    const float o_t = SIGMOID(o_ptr[j]);

                    const float c_new = f_t * cvec[j] + i_t * g_t;
                    cvec[j] = c_new;
                    hvec[j] = o_t * TANHF(c_new);
                }
            } /* batch */
        } /* layer */

        /* Salva last-layer hidden per attenzione */
        for (int b = 0; b < BATCH_SIZE; ++b) {
            float * RESTRICT dst = outputs[t][b];
            const float * RESTRICT src = st->h[NUM_LAYERS - 1][b];
            for (int j = 0; j < HIDDEN_SIZE; ++j) dst[j] = src[j];
        }
    } /* time */

    /* ---- Attenzione + FC per batch ---- */
    for (int b = 0; b < BATCH_SIZE; ++b) {
        const float * RESTRICT query = st->h[NUM_LAYERS - 1][b];

        /* score[t] = <h_t, query> con stabilizzazione per softmax */
        float score[WINDOW_SIZE];
        float max_s = -3.4028235e38f; /* -FLT_MAX */
        for (int t = 0; t < WINDOW_SIZE; ++t) {
            const float * RESTRICT ht = outputs[t][b];
            float dot = 0.f;
            for (int j = 0; j < HIDDEN_SIZE; ++j) dot += ht[j] * query[j];
            score[t] = dot;
            if (dot > max_s) max_s = dot;
        }

        /* softmax */
        float exp_sum = 0.f;
        for (int t = 0; t < WINDOW_SIZE; ++t) {
            float e = EXPPF(score[t] - max_s);
            score[t] = e;
            exp_sum += e;
        }
        const float inv_sum = 1.f / exp_sum;

        /* context = Σ alpha_t * h_t  */
        float context[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; ++j) context[j] = 0.f;
        for (int t = 0; t < WINDOW_SIZE; ++t) {
            const float a = score[t] * inv_sum;
            const float * RESTRICT ht = outputs[t][b];
            for (int j = 0; j < HIDDEN_SIZE; ++j) context[j] += a * ht[j];
        }

        /* FC head su [query | context] */
        float acc = w->fc_bias;
        const float * RESTRICT wfc = w->fc_weight;
        for (int j = 0; j < HIDDEN_SIZE; ++j) acc += wfc[j] * query[j];
        for (int j = 0; j < HIDDEN_SIZE; ++j) acc += wfc[HIDDEN_SIZE + j] * context[j];

        out[b] = acc;
    }
}

/* Piccole garanzie a compile-time */
_Static_assert(INPUT_SIZE   > 0, "INPUT_SIZE must be > 0");
_Static_assert(HIDDEN_SIZE  > 0, "HIDDEN_SIZE must be > 0");
_Static_assert(NUM_LAYERS   > 0, "NUM_LAYERS must be > 0");
_Static_assert(BATCH_SIZE   > 0, "BATCH_SIZE must be > 0");
_Static_assert(WINDOW_SIZE  > 0, "WINDOW_SIZE must be > 0");
/* ──────────────────────────────────────────────────────────────────────────── */