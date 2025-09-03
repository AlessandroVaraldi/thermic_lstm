// main.c — run inference and dump predictions/ground truth to binary files.
//
// Uso:
//   ./run_infer                 # scrive preds.bin e truth.bin nella cwd
//   ./run_infer out_preds.bin out_truth.bin
//
// Compilazione (dataset float):
//   gcc -O3 -std=c11 -Wall -Wextra -o run_infer engine.c sigmoid_poly.c tanh_poly.c main.c -lm
//
// Compilazione (dataset quantizzato TEST_XQ):
//   gcc -O3 -std=c11 -Wall -Wextra -DUSE_QTEST=1 -o run_infer engine.c sigmoid.c tanh.c main.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "engine.h"
#include "test_data.h"

static void print_examples_header(void){
    printf(" idx |   y_true   |   y_pred   |   err\n");
    printf("-----+------------+------------+-----------\n");
}

int main(int argc, char** argv){
    const char* pred_path = (argc >= 2) ? argv[1] : "preds.bin";
    const char* true_path = (argc >= 3) ? argv[2] : "truth.bin";

    FILE* fp_pred = fopen(pred_path, "wb");
    if (!fp_pred) { perror("fopen preds"); return 1; }
    FILE* fp_true = fopen(true_path, "wb");
    if (!fp_true) { perror("fopen truth"); fclose(fp_pred); return 1; }

    qlstm_state_t st;
    double sum_abs = 0.0, sum_sq = 0.0;
    const int print_k = (TEST_N < 10) ? TEST_N : 10;

    print_examples_header();

    for (int n = 0; n < TEST_N; ++n){
        qlstm_reset(&st);

        float y_pred;
        #if defined(USE_QTEST) || defined(HAS_TEST_XQ)
            y_pred = infer_window_q(TEST_XQ[n], &st);
        #else
            y_pred = infer_window(TEST_X[n], &st);
        #endif

        const float y_true = TEST_Y[n];
        const double err = (double)y_pred - (double)y_true;

        // dump in binario (float32 nativo)
        fwrite(&y_pred, sizeof(float), 1, fp_pred);
        fwrite(&y_true, sizeof(float), 1, fp_true);

        sum_abs += fabs(err);
        sum_sq  += err * err;

        if (n < print_k){
            printf("%4d | %10.4f | %10.4f | %+9.4f\n", n, y_true, y_pred, (float)err);
        }
    }

    fclose(fp_pred);
    fclose(fp_true);

    const double N = (double)TEST_N;
    const double mae  = sum_abs / N;
    const double rmse = sqrt(sum_sq / N);

    printf("-----+------------+------------+-----------\n");
    printf("MAE  = %.6f\n", mae);
    printf("RMSE = %.6f\n", rmse);
    printf("[OK] wrote %s and %s (%d float32 each)\n", pred_path, true_path, TEST_N);

    return 0;
}
