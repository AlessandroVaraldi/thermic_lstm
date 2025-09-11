// src/engine.c
#include "engine.h"
#include "acts.h"
#include "checkpoints/model_offsets.h"  // generato da json2offsets.py

// ---------- Bind pesi dal blob ----------
void qat_bind_weights(qat_model_t *m, const uint8_t *blob)
{
    // m->layers e m->fc devono essere già allocati dal chiamante (stack o static)
    for (int l=0; l<m->L; ++l) {
        lstm_layer_t *Lr = &m->layers[l];
        // offsets/gamme via macro L{l}_...
        switch (l) {
        #define BIND_L(i) do{ \
            Lr->W_ih = (const int8_t*)(blob + L##i##_IH_W_OFF); \
            Lr->B_ih = (L##i##_IH_B_NBYTES ? (const int32_t*)(blob + L##i##_IH_B_OFF) : NULL); \
            Lr->W_hh = (const int8_t*)(blob + L##i##_HH_W_OFF); \
            Lr->B_hh = (L##i##_HH_B_NBYTES ? (const int32_t*)(blob + L##i##_HH_B_OFF) : NULL); \
            Lr->in   = (int16_t)L##i##_IH_IN; \
            Lr->H    = (int16_t)(L##i##_IH_OUT/4); \
            /* rq: 0:i 1:f 2:g 3:o */ \
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
        // aggiungi se hai più layer
        default: break;
        #undef BIND_L
        }
    }
    // FC
    m->fc.W  = (const int8_t *)(blob + FC_W_OFF);
    m->fc.B  = (FC_B_NBYTES ? (const int32_t*)(blob + FC_B_OFF) : NULL);
    m->fc.in = (int16_t)FC_IN;
}

// ---------- Matvec int8 ----------
static inline int32_t dot_s8_s8_acc32(const int8_t *w_row, const int8_t *x, int len)
{
    int32_t acc = 0;
    for (int i=0;i<len;++i) acc += (int32_t)w_row[i] * (int32_t)x[i];
    return acc;
}

// ---------- Gate pack utils ----------
typedef struct { int8_t i,f,g,o; } gates_i8_t;

static inline int8_t requant_to_q8(int32_t acc, rq_q15_t rq) {
    int32_t y = mul_q15_shift(acc, rq.mult_q15, rq.rshift);
    // target pre-scale is S_pre ⇒ final in int domain ~[-128..127]
    return SAT8(y);
}

static void compute_gates_layer(const lstm_layer_t *Lr,
                                const int8_t *x_q8,   // input int8 (pre-scaled per layer)
                                const int8_t *h_q8,   // prev h converted to int8 for hh matmul
                                gates_i8_t *out)
{
    const int H = Lr->H, IN = Lr->in;
    // W_ih: [4H, IN], W_hh: [4H, H]
    const int8_t *Wih = Lr->W_ih;
    const int8_t *Whh = Lr->W_hh;
    const int32_t *Bih = Lr->B_ih, *Bhh = Lr->B_hh;

    // Per ogni gate: acc_ih→rq_ih, acc_hh→rq_hh, somma -> sat8
    for (int g=0; g<4; ++g) {
        int row = g*H; // inizio blocco gate
        int32_t acc_ih = dot_s8_s8_acc32(Wih + row*IN, x_q8, IN);
        if (Bih) acc_ih += Bih[row+0]; // bias allineato a righe (una per output)
        int8_t pre_ih = requant_to_q8(acc_ih, Lr->rq[g].ih);

        int32_t acc_hh = dot_s8_s8_acc32(Whh + row*H, h_q8, H);
        if (Bhh) acc_hh += Bhh[row+0];
        int8_t pre_hh = requant_to_q8(acc_hh, Lr->rq[g].hh);

        int16_t sum = (int16_t)pre_ih + (int16_t)pre_hh; // [-256..256]
        if (sum > 127) sum = 127; if (sum < -128) sum = -128;

        switch(g){
            case 0: out->i = (int8_t)sum; break;
            case 1: out->f = (int8_t)sum; break;
            case 2: out->g = (int8_t)sum; break;
            case 3: out->o = (int8_t)sum; break;
        }
    }
}

// h16→h_q8 helper (simple clamp, Q0.15→Q0.8 by >>7)
static inline int8_t q15_to_q8(int16_t v){ int32_t t = v >> 7; if (t>127) t=127; if (t<-128) t=-128; return (int8_t)t; }

// x_q15 (2 features) → x_q8 per layer 0: usa lo stesso >>7 (coerente con Q0.15→Q0.8)
static inline int8_t feat_q15_to_q8(int16_t v){ return q15_to_q8(v); }

// ---------- LSTM core ----------
void lstm_reset(const qat_model_t *m, int16_t *h, int16_t *c)
{
    (void)m;
    // caller alloca [L*H] per h e c
    // azzera
}

void lstm_step(const qat_model_t *m, const int16_t *x_q15, int16_t *h, int16_t *c)
{
    // Nota: qui assumiamo input D=2 (P_norm, Tbp_norm) in Q0.15 e li riduciamo a Q0.8 per il primo layer.
    int8_t x_q8_buf[256]; // sufficiente per IN<=256, adatta se serve
    int8_t h_q8_buf[256];

    // per ogni layer
    for (int l=0; l<m->L; ++l){
        const lstm_layer_t *Lr = &m->layers[l];

        const int H = Lr->H;
        // prepara x int8: layer 0 usa features, layer>0 usa h prev layer
        if (l==0){
            for (int d=0; d<Lr->in && d<256; ++d)
                x_q8_buf[d] = feat_q15_to_q8(x_q15[d]);
        } else {
            for (int d=0; d<Lr->in && d<256; ++d)
                x_q8_buf[d] = q15_to_q8(h[(l-1)*m->H + d]);
        }

        // h(l-1) → h_q8 per hh
        for (int d=0; d<H && d<256; ++d)
            h_q8_buf[d] = q15_to_q8(h[l*m->H + d]); // h_l_{t-1}

        gates_i8_t pre;
        compute_gates_layer(Lr, x_q8_buf, h_q8_buf, &pre);

        // attivazioni
        uint8_t iq = sigmoid_q8(pre.i);
        uint8_t fq = sigmoid_q8(pre.f);
        int8_t  gq = tanh_q8(pre.g);
        uint8_t oq = sigmoid_q8(pre.o);

        // update c,h: c32 = f*c >>8 + i*g >>8; tutto in Q0.15
        for (int u=0; u<H; ++u){
            int idx = l*m->H + u;
            int32_t c32 = ((int32_t)fq * (int32_t)c[idx]) >> 8;
            c32 += ((int32_t)iq * (int32_t)((int16_t)gq << 7)) >> 8; // gq Q0.8 -> Q0.15 shift <<7
            c[idx] = SAT16(c32);
            int16_t th = tanh_q15(c[idx]);
            int32_t h32 = ((int32_t)oq * (int32_t)th) >> 8;
            h[idx] = SAT16(h32);
        }
    }
}

// ---------- Attention (dot) + softmax Q15 ----------
static void softmax_q15(const int32_t *scores, int W, int16_t *w_q15)
{
    // subtract max for stability
    int32_t m = scores[0];
    for (int i=1;i<W;++i) if (scores[i] > m) m = scores[i];

    // exp approx in fixed-point: e^(x/2^S), x<=0.
    // scegli S per comprimere dinamica: qui 20 è spesso ok (tunable).
    const int SHIFT = 20;
    int64_t sum = 0;
    for (int i=0;i<W;++i){
        int32_t d = scores[i] - m; // <= 0
        int32_t z = -(d >> SHIFT); // indice_
