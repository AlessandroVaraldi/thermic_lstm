// engine.c
#include <stdint.h>
#include <string.h>
#include "engine.h"
#include "acts.h"

// Parametric frac bits for c(t)
#if C_FRAC_BITS < 8
#error "C_FRAC_BITS must be >= 8"
#endif
#define QF C_FRAC_BITS

// Row slice helpers
#define ROW_OFF(r, cols) ((size_t)(r) * (size_t)(cols))

// Dot int8 · int8 (acc int32)
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n){
  int32_t s = 0;
  for(int i=0;i<n;i++) s += (int32_t)a[i] * (int32_t)b[i];
  return s;
}

// Combine ih/hh to pre_s8 for a gate row j
static inline int8_t gate_pre_row_s8(
  const int8_t *x_in, const int8_t *h_prev,
  const int8_t *Wih, const int8_t *Whh, int IN, int H,
  int row, // absolute row in [0..4H)
  const int16_t *m_ih_q15, const int16_t *m_hh_q15,
  uint8_t rshift, int32_t bias_q
){
  const int8_t *w_ih = Wih + ROW_OFF(row, IN);
  const int8_t *w_hh = Whh + ROW_OFF(row, H);
  int32_t acc_ih = dot_i8(x_in, w_ih, IN);
  int32_t acc_hh = dot_i8(h_prev, w_hh, H);
  // align and sum (use 64-bit to avoid overflow)
  int64_t t = (int64_t)acc_ih * (int64_t)(*m_ih_q15) + (int64_t)acc_hh * (int64_t)(*m_hh_q15);
  int32_t pre = (int32_t)(t >> rshift);
  pre += bias_q;
  return sat8(pre);
}

// Convert gate outputs to Q(?,QF) from UQ8(Q0.8)/SQ8(Q1.7)
static inline int16_t uq8_to_qc(uint8_t u){ return (int16_t)((int32_t)u << (QF - 8)); }
static inline int16_t sq8_to_qc(int8_t  s){ return (int16_t)((int32_t)s << (QF - 7)); }

// h_s8 (Q1.7) <- o_q12 * tanh(c)_s8
static inline int8_t h_from_o_and_tanhc(int16_t o_qc, int8_t tanh_c_s8){
  int16_t t_qc = (int16_t)((int32_t)tanh_c_s8 << (QF - 7));   // Q1.7 -> QF
  int32_t z_qc = ( (int32_t)o_qc * (int32_t)t_qc ) >> QF;
  return sat8( z_qc >> (QF - 7) ); // QF -> Q1.7
}

// LSTM single-layer step
static void lstm_layer_step(
  const layer_t *L,
  const int8_t *x_in,   // IN dims
  const int8_t *h_prev, // H dims (s8)
  int16_t *c_state,     // H dims (Q3.12)
  int8_t  *h_out,       // H dims (s8)
  // MP inputs
  int mp_enabled, uint16_t tau_thr_q12, uint16_t delta_q12,
  const int8_t *lut_tanh_c
){
  const int H = (int)MODEL_HIDDEN;     // from model_int8.h
  const int IN = (int)L->in_dim;

  // Gate rows
  const int i0=L->row_i0, i1=L->row_i1;
  const int f0=L->row_f0, f1=L->row_f1;
  const int g0=L->row_g0, g1=L->row_g1;
  const int o0=L->row_o0, o1=L->row_o1;

  for(int j=0;j<H;j++){
    const int ri = i0 + j, rf = f0 + j, rg = g0 + j, ro = o0 + j;

    // pre-activations -> int8
    int8_t pre_i = gate_pre_row_s8(
      x_in, h_prev, L->Wih, L->Whh, IN, H,
      ri, &L->rq_i.m_ih_q15[j], &L->rq_i.m_hh_q15[j], L->rq_i.rshift[j], L->rq_i.bias_q[j]
    );
    int8_t pre_f = gate_pre_row_s8(
      x_in, h_prev, L->Wih, L->Whh, IN, H,
      rf, &L->rq_f.m_ih_q15[j], &L->rq_f.m_hh_q15[j], L->rq_f.rshift[j], L->rq_f.bias_q[j]
    );
    int8_t pre_g = gate_pre_row_s8(
      x_in, h_prev, L->Wih, L->Whh, IN, H,
      rg, &L->rq_g.m_ih_q15[j], &L->rq_g.m_hh_q15[j], L->rq_g.rshift[j], L->rq_g.bias_q[j]
    );
    int8_t pre_o = gate_pre_row_s8(
      x_in, h_prev, L->Wih, L->Whh, IN, H,
      ro, &L->rq_o.m_ih_q15[j], &L->rq_o.m_hh_q15[j], L->rq_o.rshift[j], L->rq_o.bias_q[j]
    );

    // activations via LUT
    uint8_t i_u8 = sigmoid_uq8_lut(pre_i, L->lut_sigma_i);
    uint8_t f_u8 = sigmoid_uq8_lut(pre_f, L->lut_sigma_f);
    int8_t  g_s8 = tanh_s8_lut(pre_g, L->lut_tanh_g);
    uint8_t o_u8 = sigmoid_uq8_lut(pre_o, L->lut_sigma_o);

    // to Q(?,QF)
    int16_t i_q12 = uq8_to_qc(i_u8);
    int16_t f_q12 = uq8_to_qc(f_u8);
    int16_t g_q12 = sq8_to_qc(g_s8);
    int16_t o_q12 = uq8_to_qc(o_u8);

    // c = f*c + i*g   (QF domain)
    int32_t c1 = ( (int32_t)f_q12 * (int32_t)c_state[j] ) >> QF;
    int32_t c2 = ( (int32_t)i_q12 * (int32_t)g_q12     ) >> QF;
    int32_t c_new = c1 + c2;
    // clamp to int16 (Q3.12 covers ~±8)
    c_state[j] = sat16(c_new);

    // tanh(c) with MP-time option
    int8_t tc = tanh_c_s8_mp(c_state[j], lut_tanh_c, mp_enabled, tau_thr_q12, delta_q12);

    // h = o * tanh(c) → s8
    h_out[j] = h_from_o_and_tanhc(o_q12, tc);
  }
}

// Attention over time (dot scores → softmax(q15) → context)
static void attention_context_s8(
  const int8_t *Y,   // [T*H], sequence of last-layer h(t) in s8
  const int8_t *h_T, // [H], last hidden
  int T, int H,
  int8_t *ctx_s8     // [H]
){
  int32_t scores[WIN_SIZE]; // T <= WIN_SIZE at compile-time
  int16_t w_q15[WIN_SIZE];

  // scores[t] = dot(h(t), h_T)
  for(int t=0;t<T;t++){
    const int8_t *h_t = Y + (size_t)t * (size_t)H;
    scores[t] = dot_i8(h_t, h_T, H);
  }
  // softmax-like weights (integer-only)
  softmax_pow2_q15_from_scores(scores, T, w_q15);

  // context = sum_t w[t] * h(t)
  for(int j=0;j<H;j++){
    int32_t acc_q15 = 0;
    for(int t=0;t<T;t++){
      const int8_t *h_t = Y + (size_t)t * (size_t)H;
      // promote h(t) s8 -> q15 by <<8, then mul by w_q15 (>>15)
      acc_q15 += ( (int32_t)w_q15[t] * (int32_t)((int16_t)h_t[j] << 8) ); // Q1.15 * Q1.15
    }
    // back to s8: (acc >> 15) gives q15; then >>8 → s8
    int32_t v_q15 = acc_q15 >> 15;
    ctx_s8[j] = sat8( v_q15 >> 8 );
  }
}

// FC head: z_s8 (2H) → y_s8
static inline int8_t fc_forward_s8(
  const fc_t *F, const int8_t *z_s8, int dim
){
  int32_t acc = dot_i8(z_s8, F->W, dim);
  acc += F->b[0];
  int64_t t = (int64_t)acc * (int64_t)(F->m_q15);
  int32_t y = (int32_t)(t >> F->rshift);
  return sat8(y);
}

// API: states reset to zero
void lstm_reset_states(q15_t *c, q7_t *h, uint16_t H, uint16_t L){
  (void)L;
  memset(c, 0, (size_t)H * (size_t)L * sizeof(q15_t));
  memset(h, 0, (size_t)H * (size_t)L * sizeof(q7_t));
}

// Main forward over a fixed window (normalized inputs, integer-only)
void lstm_forward_window(
  const lstm_model_t *M,
  const q7_t *x_s8,  // [Wseq*D]
  q7_t *y_s8         // (1)
){
  const int L = (int)M->L, H = (int)M->H, D = (int)M->D, T = (int)M->Wseq;

  // per-layer states
  q15_t C[LSTM_LAYERS][MODEL_HIDDEN];
  q7_t  Ht[LSTM_LAYERS][MODEL_HIDDEN];
  lstm_reset_states(&C[0][0], &Ht[0][0], M->H, M->L);

  // store last-layer sequence for attention
  int8_t Y_seq[WIN_SIZE][MODEL_HIDDEN];
  int8_t x_prev[MODEL_INPUTS] = {0};

  for(int t=0;t<T;t++){
    const int8_t *x_t = x_s8 + (size_t)t * (size_t)D;

    // MP-time trigger from |Δx_norm|
    uint16_t d_q12 = delta_q12_from_x8(x_prev, x_t, D);
    // update prev
    for(int k=0;k<D;k++) x_prev[k] = x_t[k];

    // layer 0
    const layer_t *L0 = &M->layers[0];
    lstm_layer_step(L0, x_t, Ht[0], C[0], Ht[0], M->mp_enabled, M->mp_tau_thr_q12, d_q12, M->lut_tanh_c);

    // stacked layers
    for(int li=1; li<L; li++){
      const layer_t *Lx = &M->layers[li];
      lstm_layer_step(Lx, Ht[li-1], Ht[li], C[li], Ht[li], M->mp_enabled, M->mp_tau_thr_q12, d_q12, M->lut_tanh_c);
    }
    // keep last-layer h(t)
    memcpy(Y_seq[t], Ht[L-1], (size_t)H);
  }

  // attention: query = h_T (last step)
  int8_t ctx_s8[MODEL_HIDDEN];
  attention_context_s8(&Y_seq[0][0], Ht[L-1], T, H, ctx_s8);

  // concat z = [h_T ; ctx]
  int8_t z_s8[2*MODEL_HIDDEN];
  memcpy(z_s8,           Ht[L-1], (size_t)H);
  memcpy(z_s8 + (size_t)H, ctx_s8, (size_t)H);

  // FC head
  *y_s8 = fc_forward_s8(M->fc, z_s8, 2*H);
}
