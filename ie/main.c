// main.c — confronto inference C vs PyTorch
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "include/engine.h"               // API dell’engine (fornito prima)
#include "include/acts.h"                     // attivazioni integer
#include "include/model_offsets.h"    // generato da json2offsets.py

// ======= CONFIG =======
// scegli come importare la reference PyTorch:
// 1) Header statico generato: include/ref_example.h
// 2) Binario: include/ref_example.bin
// 3) Se vuoi embed del model bin, definisci EMBED_MODEL_BIN e fornisci i simboli linker.
#define REF_FROM_HEADER 1   // 1=header, 0=binario
#define MODEL_BIN_PATH  "include/model_int8.bin"
#define REF_BIN_PATH    "include/ref_example.bin"

// ======= Opzionale: simboli linker per embed del model bin =======
// extern const unsigned char _binary_model_int8_bin_start[];
// extern const unsigned char _binary_model_int8_bin_end[];

// ------- piccolo helper per leggere un file intero in RAM -------
static unsigned char* read_file(const char* path, size_t* out_sz) {
    FILE* f = fopen(path, "rb");
    if(!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if(sz <= 0){ fclose(f); return NULL; }
    unsigned char* buf = (unsigned char*)malloc((size_t)sz);
    if(!buf){ fclose(f); return NULL; }
    size_t n = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    if(n != (size_t)sz){ free(buf); return NULL; }
    if(out_sz) *out_sz = (size_t)sz;
    return buf;
}

// -------- struttura del ref bin (Q15) --------
#pragma pack(push,1)
typedef struct {
    uint32_t magic;   // 'REF1' = 0x31464552
    uint16_t W;       // steps
    uint16_t D;       // features per step
    int16_t  y_ref_q15; // output PyTorch normalizzato in Q15
    // segue: int16_t x_q15[W*D]
} ref_bin_hdr_t;
#pragma pack(pop)

#if REF_FROM_HEADER
  #include "include/ref_example.h" // deve definire: REF_W, REF_D, const int16_t ref_x_q15[], const int16_t ref_y_q15
#endif

int main(void)
{
    // ---------- 1) Carica il blob dei pesi ----------
    const unsigned char* model_blob = NULL;
    size_t model_size = 0;

#ifdef EMBED_MODEL_BIN
    extern const unsigned char _binary_model_int8_bin_start[];
    extern const unsigned char _binary_model_int8_bin_end[];
    model_blob = _binary_model_int8_bin_start;
    model_size = (size_t)(_binary_model_int8_bin_end - _binary_model_int8_bin_start);
#else
    model_blob = read_file(MODEL_BIN_PATH, &model_size);
    if(!model_blob){
        fprintf(stderr, "[ERR] impossibile aprire %s\n", MODEL_BIN_PATH);
        return 1;
    }
#endif

    // ---------- 2) Costruisci il modello C ----------
    // Alloca i descrittori layer e il contenitore modello
    lstm_layer_t *layers = (lstm_layer_t*)calloc(MODEL_LAYERS, sizeof(lstm_layer_t));
    qat_model_t model = { .L = MODEL_LAYERS, .H = MODEL_HIDDEN, .layers = layers, .fc = {0} };
    qat_bind_weights(&model, model_blob);

    // ---------- 3) Carica la reference PyTorch ----------
    int W=0, D=0;
    int16_t *x_seq_q15 = NULL;
    int16_t y_ref_q15 = 0;

#if REF_FROM_HEADER
    W = REF_W; D = REF_D; y_ref_q15 = ref_y_q15;
    x_seq_q15 = (int16_t*)ref_x_q15; // già puntatore a dati statici
#else
    size_t ref_sz=0;
    unsigned char* ref_buf = read_file(REF_BIN_PATH, &ref_sz);
    if(!ref_buf){
        fprintf(stderr, "[ERR] impossibile aprire %s\n", REF_BIN_PATH);
        goto cleanup_err;
    }
    if(ref_sz < sizeof(ref_bin_hdr_t)){
        fprintf(stderr, "[ERR] ref bin troppo piccolo\n");
        free(ref_buf); goto cleanup_err;
    }
    ref_bin_hdr_t* hdr = (ref_bin_hdr_t*)ref_buf;
    if(hdr->magic != 0x31464552u){ // 'REF1'
        fprintf(stderr, "[ERR] magic non valido nel ref bin\n");
        free(ref_buf); goto cleanup_err;
    }
    W = hdr->W; D = hdr->D; y_ref_q15 = hdr->y_ref_q15;
    size_t expect = sizeof(ref_bin_hdr_t) + (size_t)W * (size_t)D * sizeof(int16_t);
    if(ref_sz < expect){
        fprintf(stderr, "[ERR] ref bin incompleto (atteso %zu, got %zu)\n", expect, ref_sz);
        free(ref_buf); goto cleanup_err;
    }
    x_seq_q15 = (int16_t*)(ref_buf + sizeof(ref_bin_hdr_t));
#endif

    // ---------- 4) Workspace per run ----------
    int H = model.H;
    int16_t *hseq = (int16_t*)calloc((size_t)W*(size_t)H, sizeof(int16_t));
    int16_t *h    = (int16_t*)calloc((size_t)model.L*(size_t)H, sizeof(int16_t));
    int16_t *c    = (int16_t*)calloc((size_t)model.L*(size_t)H, sizeof(int16_t));
    if(!hseq || !h || !c){
        fprintf(stderr, "[ERR] alloc workspace\n");
        goto cleanup_err;
    }

    // ---------- 5) Esegui inference C ----------
    int16_t y_c_q15 = lstm_run_sequence(&model, x_seq_q15, W, hseq, h, c);

    // ---------- 6) Confronto ----------
    int abs_err = (int)( (y_c_q15 > y_ref_q15) ? (y_c_q15 - y_ref_q15) : (y_ref_q15 - y_c_q15) );
    double yref_f = (double)y_ref_q15 / 32768.0;
    double yc_f   = (double)y_c_q15  / 32768.0;
    double rel = (y_ref_q15 != 0) ? ((double)abs_err / (double)( (y_ref_q15>=0)? y_ref_q15 : -y_ref_q15 )) : 0.0;

    printf("=== LSTM INT-only check ===\n");
    printf("W=%d, D=%d, L=%d, H=%d\n", W, D, model.L, model.H);
    printf("y_ref_q15=%d (%.6f), y_c_q15=%d (%.6f)\n", y_ref_q15, yref_f, y_c_q15, yc_f);
    printf("abs_err_q15=%d (~%.6e rel on norm)\n", abs_err, rel);

    // soglia di accettazione (tunable): 1e-3 in unità normalizzate ⇒ ~33 in Q15
    const int thr_q15 = 33;
    if(abs_err <= thr_q15){
        printf("[OK] match entro la soglia (%d <= %d)\n", abs_err, thr_q15);
    } else {
        printf("[WARN] differenza sopra soglia (%d > %d)\n", abs_err, thr_q15);
    }

    // ---------- 7) Cleanup ----------
#if !defined(EMBED_MODEL_BIN)
    free((void*)model_blob);
#endif
#if !REF_FROM_HEADER
    free((void*)((uintptr_t)x_seq_q15 - sizeof(ref_bin_hdr_t))); // ref_buf
#endif
    free(hseq); free(h); free(c); free(layers);
    return 0;

cleanup_err:
#if !defined(EMBED_MODEL_BIN)
    if(model_blob) free((void*)model_blob);
#endif
    if(layers) free(layers);
    return 1;
}
