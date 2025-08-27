#!/usr/bin/env python3
"""
Genera vettori di test per lstm_forward().

1. Crea sequenza e pesi deterministici.
2. Esegue modello equivalente in PyTorch.
3. Stampa output atteso e costanti C pronte all’uso.
"""

import torch
import numpy as np

# ───── costanti identiche a lstm_model.h ─────
INPUT_SIZE   = 2
HIDDEN_SIZE  = 4
NUM_LAYERS   = 1
BATCH_SIZE   = 2
WINDOW_SIZE  = 3

torch.manual_seed(0)
np.set_printoptions(linewidth=180,
                    formatter={'float': lambda x: f'{x:.6f}f'})

# ───── 1. sequenza deterministica ─────
seq = torch.linspace(-0.5, 0.5,
                     steps=BATCH_SIZE * WINDOW_SIZE * INPUT_SIZE,
                     dtype=torch.float32).reshape(BATCH_SIZE,
                                                  WINDOW_SIZE,
                                                  INPUT_SIZE)

# ───── 2. pesi deterministici ─────
def linspace_tensor(size, start=-1.0, end=1.0):
    return torch.linspace(start, end, steps=size, dtype=torch.float32)

W_ih = linspace_tensor(4 * HIDDEN_SIZE * INPUT_SIZE).reshape(1,
         4 * HIDDEN_SIZE, INPUT_SIZE)
W_hh = linspace_tensor(4 * HIDDEN_SIZE * HIDDEN_SIZE).reshape(1,
         4 * HIDDEN_SIZE, HIDDEN_SIZE)
b_ih = linspace_tensor(4 * HIDDEN_SIZE).reshape(1, 4 * HIDDEN_SIZE)
b_hh = torch.zeros_like(b_ih)

fc_weight = linspace_tensor(2 * HIDDEN_SIZE)          # (2H,)
fc_bias   = torch.tensor(0.1, dtype=torch.float32)    # scalare

# ───── 3. modello equivalente ─────
class SimpleAttLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(INPUT_SIZE, HIDDEN_SIZE,
                                  batch_first=True)
        self.fc   = torch.nn.Linear(2 * HIDDEN_SIZE, 1)

        with torch.no_grad():
            self.lstm.weight_ih_l0.copy_(W_ih[0])
            self.lstm.weight_hh_l0.copy_(W_hh[0])
            self.lstm.bias_ih_l0.copy_(b_ih[0])
            self.lstm.bias_hh_l0.copy_(b_hh[0])
            self.fc.weight.copy_(fc_weight.reshape(1, -1))
            self.fc.bias.copy_(fc_bias)

    def forward(self, x):
        h_seq, (h_T, _) = self.lstm(x)           # h_seq: (B,T,H)
        query   = h_T.squeeze(0)                 # (B,H)

        # dot‑product attention
        attn_logits  = (h_seq * query.unsqueeze(1)).sum(-1)    # (B,T)
        attn_weights = torch.softmax(attn_logits, dim=1)       # (B,T)
        context      = (attn_weights.unsqueeze(-1) *
                        h_seq).sum(dim=1)                      # (B,H)

        combined = torch.cat([query, context], dim=1)          # (B,2H)
        return self.fc(combined).squeeze(-1)                   # (B,)

model = SimpleAttLSTM()

with torch.no_grad():                      # <<< evita grad graph
    golden = model(seq)

# ───── 4. helper per array C ─────
def emit_c_array(name, tensor):
    flat = tensor.detach().flatten().numpy()   # <<< detach qui
    elems = ', '.join(f'{v:.6f}f' for v in flat)
    print(f"static const float {name}[{len(flat)}] = {{ {elems} }};")

print("// ---------- da copiare nel test C ----------")
emit_c_array("SEQ", seq)
emit_c_array("GOLDEN", golden)
emit_c_array("W_IH", W_ih)
emit_c_array("W_HH", W_hh)
emit_c_array("B_IH", b_ih)
emit_c_array("B_HH", b_hh)
emit_c_array("FC_WEIGHT", fc_weight)
print(f"static const float FC_BIAS = {fc_bias.item():.6f}f;")
