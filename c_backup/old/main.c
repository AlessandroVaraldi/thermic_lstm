/* ---------- main.c (multi-variant benchmark, robusto v2) ---------- */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#include "vectors.h"      /* dati generati da gen_vectors.py  */

/* ========= UTIL: timing ad alta risoluzione ========= */
static inline uint64_t now_ns(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

/* ========= Registry/descrittore modello ========= */
typedef struct {
    const char* name;

    void (*reset)(void* st);
    void (*forward)(const float* seq, void* st, const void* w, float* out);
    void (*load_weights)(void* w);

    void*  weights;
    void*  state;
    size_t weights_bytes;
    size_t state_bytes;

    int F, H, L, B, W;
    int use_static_workbuf;   /* 1 -> workbuf in .bss, 0 -> stack */
    size_t workbuf_bytes;     /* bytes buffer principale */
} Model;

/* ========= Stima MAC ========= */
static uint64_t estimate_macs(int F, int H, int L, int B, int W){
    uint64_t mac_lstm = (uint64_t)B * (uint64_t)W * (uint64_t)L * (uint64_t)(4 * H) * (uint64_t)(F + H);
    uint64_t mac_attn = (uint64_t)B * ((uint64_t)W * H + (uint64_t)W * H + (uint64_t)(2 * H));
    return mac_lstm + mac_attn;
}

/* ========= Accuracy ========= */
typedef struct { float mae, rmse, max_abs; } AccStats;

static AccStats accuracy_stats(const float* pred, const float* gold, int B){
    double sum_abs = 0.0, sum_sq = 0.0, max_abs = 0.0;
    for (int b=0; b<B; ++b){
        double d = (double)pred[b] - (double)gold[b];
        double a = fabs(d);
        sum_abs += a; sum_sq += d*d;
        if (a > max_abs) max_abs = a;
    }
    AccStats s = { (float)(sum_abs/B), (float)sqrt(sum_sq/B), (float)max_abs };
    return s;
}

/* ========= Runner ========= */
typedef struct { double mean_us, std_us; } TimeStats;

static TimeStats bench_model(Model* m, const float* seq, const float* golden,
                             int repeats, float* out_buf, AccStats* acc_out)
{
    m->load_weights(m->weights);
    m->reset(m->state);

    /* warm-up */
    m->forward(seq, m->state, m->weights, out_buf);

    /* timing */
    double sum = 0.0, sum2 = 0.0;
    for (int r=0; r<repeats; ++r){
        m->reset(m->state);
        uint64_t t0 = now_ns();
        m->forward(seq, m->state, m->weights, out_buf);
        uint64_t t1 = now_ns();
        double us = (t1 - t0) / 1000.0;
        sum += us; sum2 += us * us;
    }
    double mean = sum / repeats;
    double var  = (sum2 / repeats) - mean*mean;
    if (var < 0) var = 0;
    TimeStats ts = { .mean_us = mean, .std_us = sqrt(var) };

    *acc_out = accuracy_stats(out_buf, golden, m->B);
    return ts;
}

/* ========= Stampa ========= */
static void print_header(void){
    printf("\n%-28s | %6s | %6s | %6s | %9s | %9s | %9s | %9s | %6s\n",
           "Model", "F", "H", "W", "Weights", "State", "Workbuf", "Time(us)", "MACsM");
    printf("%-28s-+-%6s-+-%6s-+-%6s-+-%9s-+-%9s-+-%9s-+-%9s-+-%6s\n",
           "----------------------------","------","------","------","---------","---------","---------","---------","------");
}
static void print_row(const Model* m, const TimeStats* tstats, uint64_t macs){
    const double macs_M = macs / 1e6;
    printf("%-28s | %6d | %6d | %6d | %9zu | %9zu | %9zu | %9.2f | %6.1f\n",
           m->name, m->F, m->H, m->W,
           m->weights_bytes, m->state_bytes, m->workbuf_bytes,
           tstats->mean_us, macs_M);
}

/* ===================================================================================== */
/* ============================   VARIANTE 1: Reference   ============================== */
/* ===================================================================================== */

/* Rinomina funzioni static inline che potrebbero collidere */
#define fast_tanhf    ref_fast_tanhf
#define fast_expf     ref_fast_expf
#define fast_sigmoidf ref_fast_sigmoidf
/* Alias tipi/funzioni “pubbliche” */
#define LSTMModelWeights LSTMModelWeights_REF
#define LSTMState        LSTMState_REF
#define lstm_reset       lstm_reset_REF
#define lstm_forward     lstm_forward_REF
#include "lstm_model_ref.h"

/* Cattura dimensioni/macros in costanti */
enum { F_REF = INPUT_SIZE, H_REF = HIDDEN_SIZE, L_REF = NUM_LAYERS, B_REF = BATCH_SIZE, W_REF = WINDOW_SIZE };
typedef struct { float _o[WINDOW_SIZE][BATCH_SIZE][HIDDEN_SIZE]; } _workbuf_probe_REF;
enum { WB_REF = sizeof(_workbuf_probe_REF) };
#ifdef USE_STATIC_WORKBUF
enum { STATIC_WB_REF = USE_STATIC_WORKBUF };
#else
enum { STATIC_WB_REF = 0 };
#endif

/* Loader pesi — silenzia cast-qual per i campi const */
static void load_weights_REF(void* wv){
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
    LSTMModelWeights_REF* w = (LSTMModelWeights_REF*)wv;
    memcpy((void*)w->W_ih[0],   W_IH,      sizeof W_IH);
    memcpy((void*)w->W_hh[0],   W_HH,      sizeof W_HH);
    memcpy((void*)w->b_ih[0],   B_IH,      sizeof B_IH);
    memcpy((void*)w->b_hh[0],   B_HH,      sizeof B_HH);
    memcpy((void*)w->fc_weight, FC_WEIGHT, sizeof FC_WEIGHT);
    memcpy((void*)&w->fc_bias,  &FC_BIAS,  sizeof(float));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
static void reset_unified_REF(void* stv){ lstm_reset_REF((LSTMState_REF*)stv); }
static void forward_unified_REF(const float* seq, void* stv, const void* wv, float* out){
    lstm_forward_REF(seq, (LSTMState_REF*)stv, (const LSTMModelWeights_REF*)wv, out);
}
static Model make_model_REF(void){
    static LSTMModelWeights_REF W;
    static LSTMState_REF        S;
    Model m = {
        .name = "Reference (full/fast per-compile)",
        .reset = reset_unified_REF,
        .forward = forward_unified_REF,
        .load_weights = load_weights_REF,
        .weights = &W, .state = &S,
        .weights_bytes = sizeof(W), .state_bytes = sizeof(S),
        .F = F_REF, .H = H_REF, .L = L_REF, .B = B_REF, .W = W_REF,
        .use_static_workbuf = STATIC_WB_REF,
        .workbuf_bytes = WB_REF
    };
    return m;
}

/* Pulisci macro prima della prossima variante */
#undef INPUT_SIZE
#undef HIDDEN_SIZE
#undef NUM_LAYERS
#undef BATCH_SIZE
#undef WINDOW_SIZE
#undef GATE_COUNT
#undef USE_STATIC_WORKBUF
#undef USE_FAST_MATH
#undef RESTRICT
#undef LSTMModelWeights
#undef LSTMState
#undef lstm_reset
#undef lstm_forward
#undef fast_tanhf
#undef fast_expf
#undef fast_sigmoidf

/* ===================================================================================== */
/* =====================   VARIANTE 2: Mixed-Precision per-batch   ===================== */
/* ===================================================================================== */

/* Rinomina funzioni inline potenzialmente duplicate */
#define fast_tanhf    mp_fast_tanhf
#define fast_expf     mp_fast_expf
#define fast_sigmoidf mp_fast_sigmoidf
/* Alias tipi/funzioni “pubbliche” */
#define LSTMModelWeights LSTMModelWeights_MP
#define LSTMState        LSTMState_MP
#define lstm_reset       lstm_reset_MP
#define lstm_forward     lstm_forward_MP
#include "lstm_model_mp.h"

/* Cattura dimensioni/macros */
enum { F_MP = INPUT_SIZE, H_MP = HIDDEN_SIZE, L_MP = NUM_LAYERS, B_MP = BATCH_SIZE, W_MP = WINDOW_SIZE };
typedef struct { float _o[WINDOW_SIZE][BATCH_SIZE][HIDDEN_SIZE]; } _workbuf_probe_MP;
enum { WB_MP = sizeof(_workbuf_probe_MP) };
#ifdef USE_STATIC_WORKBUF
enum { STATIC_WB_MP = USE_STATIC_WORKBUF };
#else
enum { STATIC_WB_MP = 0 };
#endif

/* Loader pesi — silenzia cast-qual */
static void load_weights_MP(void* wv){
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
    LSTMModelWeights_MP* w = (LSTMModelWeights_MP*)wv;
    memcpy((void*)w->W_ih[0],   W_IH,      sizeof W_IH);
    memcpy((void*)w->W_hh[0],   W_HH,      sizeof W_HH);
    memcpy((void*)w->b_ih[0],   B_IH,      sizeof B_IH);
    memcpy((void*)w->b_hh[0],   B_HH,      sizeof B_HH);
    memcpy((void*)w->fc_weight, FC_WEIGHT, sizeof FC_WEIGHT);
    memcpy((void*)&w->fc_bias,  &FC_BIAS,  sizeof(float));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}
static void reset_unified_MP(void* stv){ lstm_reset_MP((LSTMState_MP*)stv); }
static void forward_unified_MP(const float* seq, void* stv, const void* wv, float* out){
    lstm_forward_MP(seq, (LSTMState_MP*)stv, (const LSTMModelWeights_MP*)wv, out);
}
static Model make_model_MP(void){
    static LSTMModelWeights_MP W;
    static LSTMState_MP        S;
    Model m = {
        .name = "Mixed-Precision (per-batch)",
        .reset = reset_unified_MP,
        .forward = forward_unified_MP,
        .load_weights = load_weights_MP,
        .weights = &W, .state = &S,
        .weights_bytes = sizeof(W), .state_bytes = sizeof(S),
        .F = F_MP, .H = H_MP, .L = L_MP, .B = B_MP, .W = W_MP,
        .use_static_workbuf = STATIC_WB_MP,
        .workbuf_bytes = WB_MP
    };
    return m;
}

/* Pulizia macro */
#undef INPUT_SIZE
#undef HIDDEN_SIZE
#undef NUM_LAYERS
#undef BATCH_SIZE
#undef WINDOW_SIZE
#undef GATE_COUNT
#undef USE_STATIC_WORKBUF
#undef USE_FAST_MATH
#undef RESTRICT
#undef LSTMModelWeights
#undef LSTMState
#undef lstm_reset
#undef lstm_forward
#undef fast_tanhf
#undef fast_expf
#undef fast_sigmoidf

/* ======= Aggiungere nuove varianti =======
   Ripeti lo schema: rinomina eventuali fast_*,
   alias tipi, include header, cattura macro, loader, adapter, factory, undef.
*/

/* ========= Parametri benchmark ========= */
#ifndef REPEATS
#define REPEATS 1000
#endif

#define MAX_MODELS 8

int main(void)
{
    Model models[MAX_MODELS];
    int M = 0;

    models[M++] = make_model_REF();
    models[M++] = make_model_MP();
    /* Aggiunte future:
       models[M++] = make_model_INT8();
    */

    float out[16]; /* buffer out condiviso (B piccolo) */

    print_header();

    for (int i=0; i<M; ++i){
        Model* m = &models[i];

        uint64_t macs = estimate_macs(m->F, m->H, m->L, m->B, m->W);

        AccStats acc;
        TimeStats tstats = bench_model(m, SEQ, GOLDEN, REPEATS, out, &acc);

        print_row(m, &tstats, macs);
        printf("   -> Accuracy: MAE=%.6f  RMSE=%.6f  Max|Δ|=%.6f (B=%d)\n",
               acc.mae, acc.rmse, acc.max_abs, m->B);
        printf("   -> Memory: weights=%zub  state=%zub  %s=%zub\n",
               m->weights_bytes, m->state_bytes,
               m->use_static_workbuf ? ".bss(workbuf)" : "stack(workbuf)",
               m->workbuf_bytes);
    }

    puts("\nDone.");
    return 0;
}
