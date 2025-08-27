from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as ckpt

# ==============================
# 1) Helper: fake-quant semplice
# ==============================
class FakeQuant(nn.Module):
    def __init__(self, init_scale: float = 0.05, per_tensor: bool = True, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
        self.per_tensor = per_tensor
        self.eps = eps

    @torch.no_grad()
    def update_scale_(self, x: torch.Tensor):
        maxabs = x.detach().abs().amax()
        self.scale.copy_((maxabs + self.eps) / 127.0)

    def forward(self, x: torch.Tensor):
        if self.training:
            self.update_scale_(x)
        # usa una copia disaccoppiata (stabile con checkpointing/AMP)
        s = self.scale.detach().clone().to(dtype=x.dtype, device=x.device)
        q = torch.clamp(torch.round(x / s), -127, 127)
        return q * s


# ==============================================
# 2) INT8 activations – STE con LUT (velocissime)
# ==============================================
# LUT in Q8: valori interi in [0..256] per sigmoid, [-256..256] per tanh
# indicizzate con q_x ∈ [-127..127] -> idx = q_x + 127

def _make_sigmoid_q8_lut(S_q8: int, device=None) -> torch.Tensor:
    # y_q ∈ [-127..127] → y_float_q = y_q * S_q8 / 256
    q = torch.arange(-127, 128, dtype=torch.float32, device=device)
    y = torch.sigmoid(q * (S_q8 / 256.0))
    lut = torch.round(y * 256.0).to(torch.int32)         # [0..256]
    return torch.clamp(lut, 0, 256)

def _make_tanh_q8_lut(S_q8: int, device=None) -> torch.Tensor:
    q = torch.arange(-127, 128, dtype=torch.float32, device=device)
    y = torch.tanh(q * (S_q8 / 256.0))
    lut = torch.round(y * 256.0).to(torch.int32)         # [-256..256]
    return torch.clamp(lut, -256, 256)

class SigmoidInt8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_float: torch.Tensor, S_q8: int, lut: torch.Tensor | None):
        # quantizza input nel dominio INT8 (LSB = S_q8/256)
        y_q = torch.clamp(torch.round(y_float * 256.0 / S_q8), -127, 127).to(torch.int64)
        if lut is not None:
            idx = (y_q + 127).to(torch.long)
            y_q8 = lut[idx].to(torch.int32)              # Q8 int
        else:
            # fallback (mai usato con LUT attiva)
            y = torch.sigmoid(y_float)
            y_q8 = torch.round(y * 256.0).to(torch.int32)
        y_hat = y_q8.to(y_float.dtype) / 256.0           # dequant

        # STE: usa grad della sigmoid float
        y_ref = torch.sigmoid(y_float)
        ctx.save_for_backward(y_ref)
        return y_hat

    @staticmethod
    def backward(ctx, grad_out):
        (y_ref,) = ctx.saved_tensors
        return grad_out * y_ref * (1 - y_ref), None, None

class TanhInt8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y_float: torch.Tensor, S_q8: int, lut: torch.Tensor | None):
        y_q = torch.clamp(torch.round(y_float * 256.0 / S_q8), -127, 127).to(torch.int64)
        if lut is not None:
            idx = (y_q + 127).to(torch.long)
            y_q8 = lut[idx].to(torch.int32)              # Q8 int
        else:
            y = torch.tanh(y_float)
            y_q8 = torch.round(y * 256.0).to(torch.int32)
        y_hat = y_q8.to(y_float.dtype) / 256.0

        y_ref = torch.tanh(y_float)
        ctx.save_for_backward(y_ref)
        return y_hat

    @staticmethod
    def backward(ctx, grad_out):
        (y_ref,) = ctx.saved_tensors
        return grad_out * (1 - y_ref.pow(2)), None, None

def sigmoid_int8_ste(x: torch.Tensor, S_q8: int, lut: torch.Tensor | None): 
    return SigmoidInt8STE.apply(x, S_q8, lut)

def tanh_int8_ste(x: torch.Tensor, S_q8: int, lut: torch.Tensor | None): 
    return TanhInt8STE.apply(x, S_q8, lut)

# ==============================
# 3) Linear quantizzato (QAT)
# ==============================
class QLinear(nn.Module):
    """Linear con fake-quant su input e pesi (INT8 simmetrico)."""
    def __init__(self, in_f, out_f, bias=True, init_scale_in=0.05, init_scale_w=0.02):
        super().__init__()
        self.w = nn.Parameter(torch.empty(out_f, in_f))
        self.b = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_uniform_(self.w, a=5**0.5)
        self.fq_in  = FakeQuant(init_scale_in)
        self.fq_w   = FakeQuant(init_scale_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = self.fq_in(x)              # fake-quant input
        wq = self.fq_w(self.w)          # fake-quant pesi
        return F.linear(xq, wq, self.b) # accumulo fp32 (QAT)

# =========================================
# 4) LSTM cell INT8-QAT con stato ad alta Q
#     + LUT registrate come buffer
# =========================================
class LSTMCellInt8QAT(nn.Module):
    """
    - Pesi/attivazioni QAT INT8 con fake-quant.
    - Porte: σ_int8 e tanh_int8 (STE) con scala Q8 fissata (configurabile).
    - Stato c_t in float32 (alta precisione). h_t calcolato con tanh_int8.
    - LUT per sigmoid/tanh: 3 tabelle distinte (i/f/o, g, tanh(c)).
    """
    def __init__(self, input_size: int, hidden_size: int,
                 S_gate_q8: int = 32, S_tanhc_q8: int = 64, use_lut: bool = True):
        super().__init__()
        self.H = hidden_size
        self.ih = QLinear(input_size, 4*hidden_size, bias=True)
        self.hh = QLinear(hidden_size, 4*hidden_size, bias=True)
        self.S_gate_q8  = int(S_gate_q8)
        self.S_tanhc_q8 = int(S_tanhc_q8)
        self.c_clip = 8.0
        self.use_lut = bool(use_lut)

        # --- LUT (come buffer, si muovono con .to(device)) ---
        if self.use_lut:
            # costruite su CPU; verranno spostate col .to()
            sig_gate = _make_sigmoid_q8_lut(self.S_gate_q8, device="cpu")
            tanh_gate= _make_tanh_q8_lut   (self.S_gate_q8, device="cpu")
            tanh_c   = _make_tanh_q8_lut   (self.S_tanhc_q8, device="cpu")
            self.register_buffer("lut_sig_gate", sig_gate, persistent=False)
            self.register_buffer("lut_tanh_gate", tanh_gate, persistent=False)
            self.register_buffer("lut_tanh_c",   tanh_c,   persistent=False)
        else:
            self.lut_sig_gate = None
            self.lut_tanh_gate= None
            self.lut_tanh_c   = None

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.ih(x_t) + self.hh(h)  # [B, 4H]
        i, f, g, o = gates.chunk(4, dim=1)

        i = sigmoid_int8_ste(i, self.S_gate_q8, self.lut_sig_gate)   # [0,1]
        f = sigmoid_int8_ste(f, self.S_gate_q8, self.lut_sig_gate)   # [0,1]
        g = tanh_int8_ste   (g, self.S_gate_q8, self.lut_tanh_gate)  # [-1,1]
        o = sigmoid_int8_ste(o, self.S_gate_q8, self.lut_sig_gate)   # [0,1]

        c = torch.clamp(f * c + i * g, -self.c_clip, self.c_clip)    # stato ad alta Q
        h = o * tanh_int8_ste(c, self.S_tanhc_q8, self.lut_tanh_c)   # h da tanh INT8
        return h, c

# =========================================
# 5) Encoder LSTM (unico layer) + attention
#     – TBPTT + (facoltativo) activation checkpointing
# =========================================
class LSTMModelInt8QAT(nn.Module):
    """
    Interfaccia: input (B,W,F) -> out (B,)
    Stato c_t resta float32; porte/h_t via activ INT8 (STE).
    Attention/softmax in float. Supporta:
      - tbptt_k: tronca il grafo ogni K step
      - use_ckpt: activation checkpointing nel loop temporale (sconsigliato su laptop)
      - ckpt_chunk: applica checkpointing su blocchi di passi
      - mc_dropout: abilita dropout anche in eval (per MC dropout)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.10,
                 S_gate_q8: int = 32, S_tanhc_q8: int = 64, use_lut: bool = True):
        super().__init__()
        assert num_layers == 1, "Questa implementazione supporta 1 solo layer."
        self.hidden_size = hidden_size
        self.dropout_p   = dropout_p
        self.cell = LSTMCellInt8QAT(input_size, hidden_size, S_gate_q8, S_tanhc_q8, use_lut=use_lut)
        self.fc   = nn.Linear(hidden_size * 2, 1)

    # step "puro": utile per checkpointing
    def _step(self, x_t: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        return self.cell(x_t, (h, c))

    def forward(
        self,
        x: torch.Tensor,
        *,
        mc_dropout: bool = False,
        tbptt_k: int = 0,
        use_ckpt: bool = False,
        ckpt_chunk: int = 16
    ) -> torch.Tensor:
        B, W, _ = x.shape
        device  = x.device
        h = torch.zeros(B, self.hidden_size, device=device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_size, device=device, dtype=x.dtype)

        outputs = []
        if use_ckpt:
            # (puoi lasciarlo OFF: con LUT è già veloce e risparmi recompute)
            for t in range(W):
                x_t = x[:, t, :]
                if t % max(ckpt_chunk, 1) != 0:
                    h, c = ckpt.checkpoint(self._step, x_t, h, c, use_reentrant=False)
                else:
                    h, c = self._step(x_t, h, c)
                outputs.append(h)
                if tbptt_k and ((t + 1) % tbptt_k == 0):
                    h = h.detach(); c = c.detach()
        else:
            # percorso veloce senza checkpoint
            for t in range(W):
                h, c = self._step(x[:, t, :], h, c)
                outputs.append(h)
                if tbptt_k and ((t + 1) % tbptt_k == 0):
                    h = h.detach(); c = c.detach()

        outputs = torch.stack(outputs, dim=1)   # (B, W, H)
        query   = h                              # (B, H)

        # attention: dot-product semplice
        scores  = torch.bmm(outputs, query.unsqueeze(2)).squeeze(2)   # (B, W)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1) # (B, H)

        combined = torch.cat([query, context], dim=1)                 # (B, 2H)
        combined = F.dropout(combined, p=self.dropout_p, training=(self.training or mc_dropout))
        out = self.fc(combined).squeeze(-1)                            # (B,)
        return out
