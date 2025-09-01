from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt


# =============================================
# FakeQuant with EMA observer + delay + freeze
# =============================================
class FakeQuant(nn.Module):
    """
    Fake-quantization module for QAT.
    - Per-tensor scale by default (per-channel handled explicitly in layers if needed).
    - Uses an EMA observer for max-abs to stabilize scale estimation.
    - Supports a quantization delay (bypass for the first N updates).
    - Supports freezing the scale after a given number of updates.
    """
    def __init__(
        self,
        init_scale: float = 0.05,
        per_tensor: bool = True,
        eps: float = 1e-8,
        ema_decay: float = 0.95,
        quant_delay: int = 0,
        freeze_after: int | None = None,
    ):
        super().__init__()
        self.per_tensor = per_tensor
        self.eps = eps
        self.ema_decay = float(ema_decay)
        self.quant_delay = int(max(0, quant_delay))
        self.freeze_after = int(freeze_after) if (freeze_after is not None) else None

        # buffers
        self.register_buffer("scale", torch.tensor(float(init_scale), dtype=torch.float32))
        self.register_buffer("running_maxabs", torch.tensor(127.0 * float(init_scale), dtype=torch.float32))
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("frozen", torch.tensor(0, dtype=torch.uint8))  # 0/1

    @torch.no_grad()
    def set_qat_hparams(self, *, ema_decay: float | None = None, quant_delay: int | None = None, freeze_after: int | None = None):
        if ema_decay is not None:
            self.ema_decay = float(ema_decay)
        if quant_delay is not None:
            self.quant_delay = int(max(0, quant_delay))
        if freeze_after is not None:
            self.freeze_after = int(freeze_after)

    @torch.no_grad()
    def _update_scale(self, x: torch.Tensor):
        if self.frozen.item() == 1:
            return
        maxabs = x.detach().abs().amax()
        # Initialize running_maxabs on first use to avoid cold-start bias
        if self.num_updates.item() == 0:
            self.running_maxabs.copy_(maxabs)
        else:
            m = self.ema_decay
            self.running_maxabs.copy_(m * self.running_maxabs + (1.0 - m) * maxabs)
        new_scale = torch.clamp(self.running_maxabs / 127.0, min=self.eps)
        self.scale.copy_(new_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.num_updates.add_(1)

            # Update observer (even during delay) so that scale is ready when Q starts
            self._update_scale(x)

            # Handle freeze
            if self.freeze_after is not None and self.num_updates.item() >= self.freeze_after:
                self.frozen.fill_(1)

            # Delay: bypass quantization for the first N updates
            if self.num_updates.item() <= self.quant_delay:
                return x

        # Straight-Through Estimator (STE) rounding
        s = self.scale
        q = torch.clamp(torch.round(x / s), -127.0, 127.0)
        deq = q * s
        # y = deq but with identity gradient wrt x
        return x + (deq - x).detach()


# ================================
# Quantized Linear (per-tensor)
# ================================
class QLinear(nn.Module):
    """
    Linear layer with fake-quant on inputs and weights (per-tensor scales).
    Accumulation is done in FP32; bias is FP32.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 act_init_scale: float = 0.05, w_init_scale: float = 0.05):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        # Fake-quant modules
        self.qx = FakeQuant(init_scale=act_init_scale)   # per-tensor (activations)
        self.qw = FakeQuant(init_scale=w_init_scale)     # per-tensor (weights)

        # init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = self.qx(x) if self.training else x
        wq = self.qw(self.weight) if self.training else self.weight
        return F.linear(xq, wq, self.bias)


# =====================================
# INT8-like activations with STE output
# =====================================
class SigmoidInt8STE(nn.Module):
    """
    Sigmoid followed by Q8 fake-quantization on the output (0..1 mapped to 0..256).
    S_gate_q8 is kept for API compatibility (used at deploy to build LUTs), but
    here we quantize the *output* to 1/256 steps with STE.
    """
    def __init__(self, S_gate_q8: int = 64):
        super().__init__()
        self.S_gate_q8 = int(S_gate_q8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(x)
        q = torch.clamp(torch.round(y * 256.0), 0.0, 256.0)
        deq = q / 256.0
        return y + (deq - y).detach()


class TanhInt8STE(nn.Module):
    """
    Tanh followed by Q8 fake-quantization on the output (-1..1 mapped to -256..256).
    S_tanhc_q8 is kept for API compatibility (deploy LUTs); training uses STE on output.
    """
    def __init__(self, S_tanhc_q8: int = 128):
        super().__init__()
        self.S_tanhc_q8 = int(S_tanhc_q8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(x)
        q = torch.clamp(torch.round(y * 256.0), -256.0, 256.0)
        deq = q / 256.0
        return y + (deq - y).detach()


# ===============================
# QAT LSTM cell (INT8 activations)
# ===============================
class QLSTMCell(nn.Module):
    """
    LSTM cell with:
      - QLinear for input/hidden projections (fake-quantized weights/acts)
      - INT8-like gate activations with STE (σ, tanh)
      - Cell state kept in FP32 (with optional clamp to [-8,8] to avoid blow-up)
    """
    def __init__(self, input_size: int, hidden_size: int,
                 S_gate_q8: int = 64, S_tanhc_q8: int = 128):
        super().__init__()
        H = hidden_size
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)

        self.ih = QLinear(input_size, 4 * H, bias=True)
        self.hh = QLinear(H,         4 * H, bias=True)

        self.sigmoid_q8 = SigmoidInt8STE(S_gate_q8)
        self.tanh_q8    = TanhInt8STE(S_tanhc_q8)

        self.c_clamp = 8.0  # mildly constrain c to stabilize long sequences

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.ih(x_t) + self.hh(h)  # (B,4H)
        i, f, g, o = gates.chunk(4, dim=1)
        i = self.sigmoid_q8(i)
        f = self.sigmoid_q8(f)
        g = self.tanh_q8(g)
        o = self.sigmoid_q8(o)
        c = f * c + i * g
        # Optional clamp to keep c in a range that matches LUT-based tanh later
        if self.c_clamp is not None:
            c = torch.clamp(c, -self.c_clamp, self.c_clamp)
        h = o * self.tanh_q8(c)
        return h, c


# ===========================================
# High-level model: LSTM + attention + head
# ===========================================
class LSTMModelInt8QAT(nn.Module):
    """
    Quantization-aware, physics-ready LSTM model for T̂ prediction.
    - Input:  (B, W, D)
    - Output: (B,) scalar (°C normalized; de-norm handled outside)
    - Internals: stacked QLSTM + simple dot-product attention on the last layer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout_p: float = 0.0,
        S_gate_q8: int = 64,
        S_tanhc_q8: int = 128,
    ):
        super().__init__()
        assert num_layers >= 1
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_p = float(dropout_p)

        layers = []
        in_sz = input_size
        for _ in range(num_layers):
            layers.append(QLSTMCell(in_sz, hidden_size, S_gate_q8, S_tanhc_q8))
            in_sz = hidden_size
        self.layers = nn.ModuleList(layers)

        # attention + head
        self.fc = QLinear(2 * hidden_size, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,                          # (B, W, D)
        use_ckpt: bool = False,
        ckpt_chunk: int = 0,
        tbptt_k: int = 0,
        mc_dropout: bool = False,
    ) -> torch.Tensor:
        B, W, D = x.shape
        assert D == self.input_size

        # initial states
        h_states = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
        c_states = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]

        outputs = []
        # Simple unroll; kwargs are accepted to enable upstream capability detection,
        # but here we keep it straightforward for robustness/speed.
        for t in range(W):
            x_t = x[:, t, :]
            for l, cell in enumerate(self.layers):
                h_states[l], c_states[l] = cell(x_t, (h_states[l], c_states[l]))
                x_t = h_states[l]
                # TBPTT detach (optional, if requested)
                if tbptt_k and ((t + 1) % int(tbptt_k) == 0):
                    h_states[l] = h_states[l].detach()
                    c_states[l] = c_states[l].detach()
            outputs.append(x_t)

        # attention over time
        H_last = h_states[-1]  # (B,H)
        Y_seq = torch.stack(outputs, dim=1)  # (B,W,H)
        scores = torch.bmm(Y_seq, H_last.unsqueeze(2)).squeeze(2)  # (B,W)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), Y_seq).squeeze(1)  # (B,H)

        combined = torch.cat([H_last, context], dim=1)  # (B,2H)
        combined = F.dropout(combined, p=self.dropout_p, training=(self.training or mc_dropout))
        out = self.fc(combined).squeeze(-1)  # (B,)
        return out
