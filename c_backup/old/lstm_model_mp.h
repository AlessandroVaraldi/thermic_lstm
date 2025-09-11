/* ─────────────────────── lstm_model.h (MCU-optimized + MP per-batch) ─────────────────────── */
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

/* Work buffer policy: 0 = on stack (default), 1 = static (zero-init in .bss) */
#ifndef USE_STATIC_WORKBUF
#define USE_STATIC_WORKBUF 0
#endif

/* Fast-math policy: 1 = inline approx (riduce code-size / call overhead) */
#ifndef USE_FAST_MATH
#define USE_FAST_MATH 1
#endif

/* Hint per optimizer (C99) */
#ifndef RESTRICT
#define RESTRICT restrict
#endif

/*========================  MIXED-PRECISION TEMPORALE (MP)  ========================*/
/* Soglie/parametri di default: adattali ai tuoi dati (unità = media |Δx| per passo) */
#ifndef MP_TAU_UP
#define MP_TAU_UP          0.05f   /* soglia salita → entra in L1 (TRANSIENT) */
#endif
#ifndef MP_TAU_DOWN
#define MP_TAU_DOWN        0.025f  /* soglia discesa → ritorna a L0 (STEADY)  */
#endif
#ifndef MP_COOLDOWN_STEPS
#define MP_COOLDOWN_STEPS  8       /* passi minimi da restare in L1 prima del ritorno */
#endif
#ifndef MP_EMA_ALPHA
#define MP_EMA_ALPHA       0.125f  /* EMA per |Δx| (≈ 1/8) */
#endif
#ifndef MP_SENTINEL_K
#define MP_SENTINEL_K      0       /* 0 = disabilitato; >0 forza 1 passo in L1 ogni K */
#endif
#ifndef MP_SOFTMAX_MODE
/* Softmax di default in L1 per robustezza; cambia in MP_STEADY per farla “mista” */
#define MP_SOFTMAX_MODE    1 /* 0=STEADY(L0-fast), 1=TRANSIENT(L1-full) */
#endif

typedef enum { MP_STEADY = 0, MP_TRANSIENT = 1 } mp_mode_t;

typedef struct {
    mp_mode_t mode;
    uint8_t   cooldown;
    float     tau_up, tau_down;
    float     ema_dx;       /* EMA della media |x_t - x_{t-1}| per questo batch */
    uint16_t  sentinel_K;   /* 0=off; altrimenti forza L1 ogni K passi */
} MPController;

static inline void mp_init(MPController* c){
    c->mode       = MP_TRANSIENT;     /* prudente al primo passo */
    c->cooldown   = 0;
    c->tau_up     = MP_TAU_UP;
    c->tau_down   = MP_TAU_DOWN;
    c->ema_dx     = 0.f;
    c->sentinel_K = MP_SENTINEL_K;
}

static inline void mp_update(MPController* c, float stat_dx){
    /* EMA cheap */
    c->ema_dx = (1.f - MP_EMA_ALPHA) * c->ema_dx + MP_EMA_ALPHA * stat_dx;

    if (c->mode == MP_STEADY){
        if (c->ema_dx > c->tau_up) { c->mode = MP_TRANSIENT; c->cooldown = MP_COOLDOWN_STEPS; }
    } else { /* TRANSIENT */
        if (c->cooldown > 0) c->cooldown--;
        if (c->ema_dx < c->tau_down && c->cooldown == 0) c->mode = MP_STEADY;
    }
}

/* Derivata media per un batch specifico (b): |x_t - x_{t-1}| media sulle F feature */
static inline float mp_dx_mean_step_batch(const float* RESTRICT seq, int t, int b){
    if (t <= 0) return 0.f;
    const int strideBW = WINDOW_SIZE * INPUT_SIZE;
    const float* cur  = seq + b*strideBW + t    *INPUT_SIZE;
    const float* prev = seq + b*strideBW + (t-1)*INPUT_SIZE;
    float sum = 0.f;
    for (int i = 0; i < INPUT_SIZE; ++i){
        float d = cur[i] - prev[i];
        sum += (d >= 0.f ? d : -d);
    }
    return sum / (float)INPUT_SIZE;
}

/*------- Math utils -------*/
#if USE_FAST_MATH
/* Polinomiale stabile/cheap per tanh, err ~1e-3 su |x|<=5 */
static inline float fast_tanhf(float x){
    const float x2 = x * x;
    return x * (27.f + x2) / (27.f + 9.f * x2);
}
static inline float fast_expf(float x){
    /* exp(x) ≈ 2^(x/ln2) con correzione quadratica */
    const float inv_ln2 = 1.4426950408889634f;   /* 1/ln(2) */
    float y = x * inv_ln2;
    int   yi = (int) (y > 0 ? y + 0.5f : y - 0.5f);
    float r = (y - yi) * 0.6931471805599453f;    /* (y-yi)*ln2 */
    float er = 1.f + r + 0.5f * r * r;           /* exp(r) ~ 1 + r + r^2/2 */
    return ldexpf(er, yi);
}
static inline float fast_sigmoidf(float x){ return 0.5f * (1.f + fast_tanhf(0.5f * x)); }
#endif

/* Wrapper run-time: selezionano fast(L0) o full(L1) */
static inline float mp_tanh_sel(float x, mp_mode_t m){
#if USE_FAST_MATH
    return (m == MP_STEADY) ? fast_tanhf(x) : tanhf(x);
#else
    (void)m; return tanhf(x);
#endif
}
static inline float mp_expf_sel(float x, mp_mode_t m){
#if USE_FAST_MATH
    return (m == MP_STEADY) ? fast_expf(x)  : expf(x);
#else
    (void)m; return expf(x);
#endif
}
static inline float mp_sigmoid_sel(float x, mp_mode_t m){
#if USE_FAST_MATH
    return (m == MP_STEADY) ? fast_sigmoidf(x) : (1.f / (1.f + expf(-x)));
#else
    (void)m; return 1.f / (1.f + expf(-x));
#endif
}

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
    /* Controller Mixed-Precision per-batch */
    MPController mpc[BATCH_SIZE];
    for (int b = 0; b < BATCH_SIZE; ++b) mp_init(&mpc[b]);

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
                /* Aggiorna controller per-batch (statistica locale) */
                if (t > 0) {
                    float s_b = mp_dx_mean_step_batch(seq, t, b);
                    mp_update(&mpc[b], s_b);
                }
                /* Seleziona il modo per questo (t,b); sentinel opzionale */
                mp_mode_t step_mode = mpc[b].mode;
                if (mpc[b].sentinel_K > 0 && ((t % mpc[b].sentinel_K) == 0))
                    step_mode = MP_TRANSIENT;

                const float * RESTRICT x_b_t = x_t0 + b * (WINDOW_SIZE * INPUT_SIZE);
                const float * RESTRICT inp   = (l == 0) ? x_b_t : st->h[l-1][b];

                /* Gate pre-activations: y = W_ih*x + W_hh*h_prev + b */
                float gates[GATE_COUNT];
                for (int g = 0; g < GATE_COUNT; ++g) {
                    const float * RESTRICT wih = W_ih + g * INPUT_SIZE;
                    const float * RESTRICT whh = W_hh + g * HIDDEN_SIZE;

                    float acc = b_sum[g]; /* bias pre-sommato */

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
                    const float i_t = mp_sigmoid_sel(i_ptr[j], step_mode);
                    const float f_t = mp_sigmoid_sel(f_ptr[j], step_mode);
                    const float g_t = mp_tanh_sel   (g_ptr[j], step_mode);
                    const float o_t = mp_sigmoid_sel(o_ptr[j], step_mode);

                    const float c_new = f_t * cvec[j] + i_t * g_t;
                    cvec[j] = c_new;
                    hvec[j] = o_t * mp_tanh_sel(c_new, step_mode);
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

        /* softmax (policy selezionabile: default L1 per robustezza) */
        float exp_sum = 0.f;
        for (int t = 0; t < WINDOW_SIZE; ++t) {
            float e = mp_expf_sel(score[t] - max_s, (mp_mode_t)MP_SOFTMAX_MODE);
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
/* ─────────────────────────────────────────────────────────────────────────────────────────── */
