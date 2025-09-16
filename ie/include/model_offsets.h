// Auto-generated: model_offsets.h (no JSON required)
#pragma once
#include <stdint.h>

#define MODEL_LAYERS   1
#define MODEL_HIDDEN   ((int)16 )
#define MODEL_INPUT    2
#define MODEL_WIN      64

#define NORM_AX0_Q15   5320
#define NORM_BX0_Q15  -4802
#define NORM_AX1_Q15   2277
#define NORM_BX1_Q15  -32767
#define NORM_AY_Q15    1683
#define NORM_BY_Q15   -32767

// Layer 0
#define L0_S_GATE_Q8   32
#define L0_S_TANHC_Q8  64
#define L0_IH_W_OFF    0u
#define L0_IH_W_NBYTES 128u
#define L0_IH_B_OFF    128u
#define L0_IH_B_NBYTES 256u
#define L0_IH_OUT      64
#define L0_IH_IN       2
#define L0_HH_W_OFF    384u
#define L0_HH_W_NBYTES 1024u
#define L0_HH_B_OFF    1408u
#define L0_HH_B_NBYTES 256u
#define L0_HH_OUT      64
#define L0_HH_IN       16
#define L0_IH_RQ_M_i   17695
#define L0_IH_RQ_S_i   18
#define L0_HH_RQ_M_i   17695
#define L0_HH_RQ_S_i   18
#define L0_IH_RQ_M_f   18854
#define L0_IH_RQ_S_f   18
#define L0_HH_RQ_M_f   18854
#define L0_HH_RQ_S_f   18
#define L0_IH_RQ_M_g   20637
#define L0_IH_RQ_S_g   18
#define L0_HH_RQ_M_g   20637
#define L0_HH_RQ_S_g   18
#define L0_IH_RQ_M_o   25914
#define L0_IH_RQ_S_o   18
#define L0_HH_RQ_M_o   25914
#define L0_HH_RQ_S_o   18

// FC head
#define FC_W_OFF       1664u
#define FC_W_NBYTES    32u
#define FC_B_OFF       1696u
#define FC_B_NBYTES    4u
#define FC_OUT         1
#define FC_IN          32
#define FC_RQ_M        20972
#define FC_RQ_S        8