/* ---------- main.c (aggiornato) ---------- */
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "lstm_model.h"   /* versione MCU‑friendly */
#include "vectors.h"      /* dati generati da gen_vectors.py */

#define EPS 1e-5f

int main(void)
{
    /* ── 1. Costruisci la struttura pesi ─────────────────────── */
    /*  NB: in  lstm_model.h i campi sono `const float`, quindi   *
     *  dobbiamo castare il puntatore di destinazione a `void *` */
    LSTMModelWeights w = {0};

    memcpy((void*)w.W_ih[0],   W_IH,      sizeof W_IH);
    memcpy((void*)w.W_hh[0],   W_HH,      sizeof W_HH);
    memcpy((void*)w.b_ih[0],   B_IH,      sizeof B_IH);
    memcpy((void*)w.b_hh[0],   B_HH,      sizeof B_HH);
    memcpy((void*)w.fc_weight, FC_WEIGHT, sizeof FC_WEIGHT);
    memcpy((void*)&w.fc_bias,  &FC_BIAS,  sizeof(float));
    
    /* ── 2. Inizializza lo stato LSTM (h, c) ─────────────────── */
    LSTMState st;
    lstm_reset(&st);                 /* azzera h e c */

    /* ── 3. Esegui la forward ────────────────────────────────── */
    float out[BATCH_SIZE];
    lstm_forward(SEQ, &st, &w, out); /* nuova firma: seq, state, w, out */

    /* ── 4. Verifica i risultati ─────────────────────────────── */
    int ok = 1;
    for (int b = 0; b < BATCH_SIZE; ++b) {
        float diff = fabsf(out[b] - GOLDEN[b]);
        if (diff > EPS) {
            printf("✗ batch %d: got %.7f, expected %.7f (Δ=%.1e)\n",
                   b, out[b], GOLDEN[b], diff);
            ok = 0;
        }
    }

    puts(ok ? "✓ lstm_forward OK" : "✗ mismatch");
    return ok ? 0 : 1;
}
