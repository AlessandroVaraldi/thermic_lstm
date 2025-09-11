#!/usr/bin/env python3
import json, sys, pathlib
j = json.loads(pathlib.Path("checkpoints/model_int8.json").read_text())
L = len(j["layers"])

out = []
W = out.append
W("// Auto-generated from model_int8.json")
W("#pragma once")
W("#include <stdint.h>")
W("")
W(f"#define MODEL_LAYERS {L}")
W(f"#define MODEL_HIDDEN ( (int){ j['layers'][0]['ih']['W_shape'][0] // 4 } )")
W("")
# offsets (in byte) e shapes
for li,layer in enumerate(j["layers"]):
    o = j["offsets"]["layers"][li]
    ihs = layer["ih"]["W_shape"]; hhs = layer["hh"]["W_shape"]
    W(f"// Layer {li}")
    W(f"#define L{li}_IH_W_OFF     {o['ih_W_off']}u")
    W(f"#define L{li}_IH_W_NBYTES  {o['ih_W_nbytes']}u")
    if 'ih_b_off' in o:
        W(f"#define L{li}_IH_B_OFF     {o['ih_b_off']}u")
        W(f"#define L{li}_IH_B_NBYTES  {o['ih_b_nbytes']}u")
    else:
        W(f"#define L{li}_IH_B_OFF     0u")
        W(f"#define L{li}_IH_B_NBYTES  0u")
    W(f"#define L{li}_IH_OUT        {ihs[0]}")
    W(f"#define L{li}_IH_IN         {ihs[1]}")
    W(f"#define L{li}_HH_W_OFF     {o['hh_W_off']}u")
    W(f"#define L{li}_HH_W_NBYTES  {o['hh_W_nbytes']}u")
    if 'hh_b_off' in o:
        W(f"#define L{li}_HH_B_OFF     {o['hh_b_off']}u")
        W(f"#define L{li}_HH_B_NBYTES  {o['hh_b_nbytes']}u")
    else:
        W(f"#define L{li}_HH_B_OFF     0u")
        W(f"#define L{li}_HH_B_NBYTES  0u")
    W(f"#define L{li}_HH_OUT        {hhs[0]}")
    W(f"#define L{li}_HH_IN         {hhs[1]}")
    rq_ih = layer["ih"]["requant"]; rq_hh = layer["hh"]["requant"]
    for g,gi in zip(("i","f","g","o"), range(4)):
        W(f"#define L{li}_IH_RQ_M_{g} {rq_ih[g]['mult_q15']}")
        W(f"#define L{li}_IH_RQ_S_{g} {rq_ih[g]['rshift']}")
        W(f"#define L{li}_HH_RQ_M_{g} {rq_hh[g]['mult_q15']}")
        W(f"#define L{li}_HH_RQ_S_{g} {rq_hh[g]['rshift']}")
    W("")
# FC
fc = j["fc"]; ofc = j["offsets"]["fc"]
W("// FC head")
W(f"#define FC_W_OFF {ofc['W_off']}u")
W(f"#define FC_W_NBYTES {ofc['W_nbytes']}u")
if "b_off" in ofc:
    W(f"#define FC_B_OFF {ofc['b_off']}u")
    W(f"#define FC_B_NBYTES {ofc['b_nbytes']}u")
else:
    W(f"#define FC_B_OFF 0u")
    W(f"#define FC_B_NBYTES 0u")
W(f"#define FC_OUT {fc['W_shape'][0]}")
W(f"#define FC_IN  {fc['W_shape'][1]}")
pathlib.Path("ie/include/model_offsets.h").write_text("\n".join(out))
print("Wrote model_offsets.h")
