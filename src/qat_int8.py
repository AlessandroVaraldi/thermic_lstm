from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from contextlib import contextmanager

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
        self.force_quant: bool = False  # abilita quant anche in eval (usato da quant_eval)

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
        do_q = self.training or self.force_quant
        xq = self.qx(x) if do_q else x
        wq = self.qw(self.weight) if do_q else self.weight
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
        self._grid = 256.0  # passi di quantizzazione sull'output (non toccata dalla MP-time)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.sigmoid(x)
        q = torch.clamp(torch.round(y * self._grid), 0.0, self._grid)
        deq = q / self._grid
        return y + (deq - y).detach()


class TanhInt8STE(nn.Module):
    """
    Tanh followed by Q8 fake-quantization on the output (-1..1 mapped to -256..256).
    S_tanhc_q8 is kept for API compatibility (deploy LUTs); training uses STE on output.
    """
    def __init__(self, S_tanhc_q8: int = 128):
        super().__init__()
        self.S_tanhc_q8 = int(S_tanhc_q8)
        self._grid = 256.0  # passi di quantizzazione sull'output (base); può essere aumentata per MP-time

    @torch.no_grad()
    def set_grid(self, grid: float):
        """Imposta la griglia di quantizzazione output (>=256 => risoluzione più fine)."""
        self._grid = float(max(1.0, grid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.tanh(x)
        q = torch.clamp(torch.round(y * self._grid), -self._grid, self._grid)
        deq = q / self._grid
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
        # separiamo tanh per g e per c(t) per abilitare MP-time solo su tanh(c)
        self.tanh_q8_g  = TanhInt8STE(S_tanhc_q8)
        self.tanh_q8_c  = TanhInt8STE(S_tanhc_q8)

        self.c_clamp = 8.0  # mildly constrain c to stabilize long sequences

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.ih(x_t) + self.hh(h)  # (B,4H)
        i, f, g, o = gates.chunk(4, dim=1)
        i = self.sigmoid_q8(i)
        f = self.sigmoid_q8(f)
        g = self.tanh_q8_g(g)
        o = self.sigmoid_q8(o)
        c = f * c + i * g
        # Optional clamp to keep c in a range that matches LUT-based tanh later
        if self.c_clamp is not None:
            c = torch.clamp(c, -self.c_clamp, self.c_clamp)
        h = o * self.tanh_q8_c(c)
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
        # ----- quant-eval toggle -----
        self._quant_eval_enabled: bool = False

        # ----- mixed-precision temporale (leggera) -----
        self._mp_time_enabled: bool = False
        self._mp_tau_thr: float = 0.08
        self._mp_grid_base: float = 256.0
        self._mp_grid_fine: float = 512.0  # default; viene settata da enable_time_mixed_precision()

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
        prev_x = None
        # Simple unroll; kwargs are accepted to enable upstream capability detection,
        # but here we keep it straightforward for robustness/speed.
        for t in range(W):
            x_t = x[:, t, :]

            # ------ MP-time: stima transiente semplice dal delta ingresso ------
            use_fine = False
            if self._mp_time_enabled and (prev_x is not None):
                # tau_t = max |Δx| su batch (coarse e cheap per evitare overhead)
                tau_t = (x_t - prev_x).abs().amax().item()
                use_fine = (tau_t > self._mp_tau_thr)
                # imposta griglia solo per tanh(c) di TUTTI i layer a questo timestep
                grid = self._mp_grid_fine if use_fine else self._mp_grid_base
                for cell in self.layers:
                    cell.tanh_q8_c.set_grid(grid)
            prev_x = x_t

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
    
    # ------------- Quant eval: lascia attive le fake-quant anche in eval -------------
    @contextmanager
    def quant_eval(self, enabled: bool = True):
        """Mantiene attiva la fake-quant anche in eval() per validazione 'INT-like'."""
        if not enabled:
            yield
            return
        prev = self._quant_eval_enabled
        self._quant_eval_enabled = True
        # Propaga a tutte le QLinear
        prev_flags = []
        for m in self.modules():
            if isinstance(m, QLinear):
                prev_flags.append((m, m.force_quant))
                m.force_quant = True
        try:
            yield
        finally:
            for m, flag in prev_flags:
                m.force_quant = flag
            self._quant_eval_enabled = prev

    # ------------- MP-time: setup leggero (solo griglia tanh(c)) -------------
    @torch.no_grad()
    def enable_time_mixed_precision(self, *, tau_thr: float = 0.08, scale_mul: float = 1.5, rshift_delta: int = -1):
        """
        Abilita una MP-temporale leggera aumentando la griglia di quantizzazione SOLO per tanh(c)
        quando |Δx| supera la soglia. Nessun cambio di attivazioni, costo trascurabile.
          - tau_thr: soglia su max|Δx| per timestep (coarse, batch-wide).
          - scale_mul: fattore moltiplicativo della griglia (>=1).
          - rshift_delta: delta shift simulato (negativo => più fine => x2^(-delta)).
        """
        self._mp_time_enabled = True
        self._mp_tau_thr = float(tau_thr)
        base = 256.0
        grid_mul = max(1.0, float(scale_mul)) * (2.0 ** float(-int(rshift_delta)))
        self._mp_grid_base = base
        self._mp_grid_fine = base * grid_mul

    # ------------- Export quant metadata (leggero) -------------
    @torch.no_grad()
    def export_quant_metadata(self) -> dict:
        """
        Raccoglie scale per-tensor da QLinear (attivazioni e pesi) + info attivazioni.
        Utile per costruire header C; NON calcola mult/rshift (dipende dal tuo toolchain).
        """
        cells = []
        for i, cell in enumerate(self.layers):
            cells.append({
                "idx": i,
                "ih": {"Sx": float(cell.ih.qx.scale), "Sw": float(cell.ih.qw.scale), "bias_fp32": bool(cell.ih.bias is not None)},
                "hh": {"Sx": float(cell.hh.qx.scale), "Sw": float(cell.hh.qw.scale), "bias_fp32": bool(cell.hh.bias is not None)},
                "S_gate_q8": int(cell.sigmoid_q8.S_gate_q8),
                "S_tanhc_q8": int(cell.tanh_q8_c.S_tanhc_q8),
            })
        meta = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "fc": {"Sx": float(self.fc.qx.scale), "Sw": float(self.fc.qw.scale), "bias_fp32": bool(self.fc.bias is not None)},
            "cells": cells,
            "mp_time": {
                "enabled": bool(self._mp_time_enabled),
                "tau_thr": float(self._mp_tau_thr),
                "grid_base": float(self._mp_grid_base),
                "grid_fine": float(self._mp_grid_fine),
            },
        }
        return meta

    @torch.no_grad()
    def emit_c_header(self, qmeta: dict | None = None) -> str:
        """
        Genera un header C minimale con define e scale (da integrare col tuo exporter pesi).
        """
        if qmeta is None:
            qmeta = self.export_quant_metadata()
        lines = []
        push = lines.append
        push("// Auto-generated quant header (minimal)")
        push("#pragma once")
        push("")
        push(f"#define MODEL_INPUT_SIZE   {qmeta['input_size']}")
        push(f"#define MODEL_HIDDEN_SIZE  {qmeta['hidden_size']}")
        push(f"#define MODEL_NUM_LAYERS   {qmeta['num_layers']}")
        push("")
        push("// FC per-tensor scales")
        push(f"static const float FC_SX = {qmeta['fc']['Sx']}f;")
        push(f"static const float FC_SW = {qmeta['fc']['Sw']}f;")
        push("")
        push("// LSTM cells per-layer scales")
        for c in qmeta["cells"]:
            i = c["idx"]
            push(f"// Layer {i}")
            push(f"static const float L{i}_IH_SX = {c['ih']['Sx']}f;")
            push(f"static const float L{i}_IH_SW = {c['ih']['Sw']}f;")
            push(f"static const float L{i}_HH_SX = {c['hh']['Sx']}f;")
            push(f"static const float L{i}_HH_SW = {c['hh']['Sw']}f;")
            push(f"static const int   L{i}_S_GATE_Q8  = {c['S_gate_q8']};")
            push(f"static const int   L{i}_S_TANHC_Q8 = {c['S_tanhc_q8']};")
            push("")
        push("// Mixed-precision temporale (griglie tanh(c))")
        push(f"#define MP_TIME_ENABLED  {1 if qmeta['mp_time']['enabled'] else 0}")
        push(f"static const float TANHC_GRID_BASE = {qmeta['mp_time']['grid_base']}f;")
        push(f"static const float TANHC_GRID_FINE = {qmeta['mp_time']['grid_fine']}f;")
        push("")
        return "\n".join(lines)
    
    @torch.no_grad()
    def calibrate_preact_scales(self, dl, device, max_batches=8):
        """
        Pass veloce su dl per stimare gli amax dei pre-attivazioni gate per layer.
        Restituisce: list[dict] con chiavi 'i','f','g','o' -> amax float.
        Usa i pesi quantizzati (quant_eval) se possibile.
        """
        # init running amax
        stats = [{"i":0.0,"f":0.0,"g":0.0,"o":0.0} for _ in range(self.num_layers)]
        def _update(layer_idx, i,f,g,o):
            s = stats[layer_idx]
            s["i"] = max(s["i"], float(i.detach().abs().amax().cpu()))
            s["f"] = max(s["f"], float(f.detach().abs().amax().cpu()))
            s["g"] = max(s["g"], float(g.detach().abs().amax().cpu()))
            s["o"] = max(s["o"], float(o.detach().abs().amax().cpu()))

        # leggero hook locale: ricalcola gates dentro
        from contextlib import nullcontext
        cm = self.quant_eval(True) if hasattr(self, "quant_eval") else nullcontext()
        with cm:
            self.eval()
            n = 0
            for xb, yb in dl:
                xb = xb.to(device)
                B, W, _ = xb.shape
                # stati
                h = [xb.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
                c = [xb.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
                for t in range(W):
                    x_t = xb[:,t,:]
                    for L, cell in enumerate(self.layers):
                        gates = cell.ih(x_t) + cell.hh(h[L])  # (B,4H)
                        i,f,g,o = gates.chunk(4, dim=1)
                        _update(L, i,f,g,o)
                        # step normale
                        i = cell.sigmoid_q8(i); f = cell.sigmoid_q8(f); g = cell.tanh_q8_g(g); o = cell.sigmoid_q8(o)
                        c[L] = f * c[L] + i * g
                        if cell.c_clamp is not None:
                            c[L] = torch.clamp(c[L], -cell.c_clamp, cell.c_clamp)
                        h[L] = o * cell.tanh_q8_c(c[L])
                        x_t  = h[L]
                n += 1
                if n >= max_batches: break

        # converti in scale per int8 simmetriche (zero-point 0)
        pre_scales = []
        for s in stats:
            pre_scales.append({k: (max(1e-8, s[k])/127.0) for k in ("i","f","g","o")})
        return pre_scales

    @torch.no_grad()
    def quantize_linear_int8(self, lin: nn.Module):
        """Ritorna (W_s8, bias_i32, Sx, Sw) usando le scale dei FakeQuant del layer."""
        Sx = float(lin.qx.scale); Sw = float(lin.qw.scale)
        W  = torch.clamp(torch.round(lin.weight / Sw), -127, 127).to(torch.int8).contiguous()
        if lin.bias is not None:
            b_i32 = torch.round(lin.bias / (Sx*Sw)).to(torch.int32).contiguous()
        else:
            b_i32 = torch.zeros(lin.out_features, dtype=torch.int32)
        return W, b_i32, Sx, Sw

    @torch.no_grad()
    def compute_requant(self, Sx: float, Sw: float, S_pre: float, *, max_shift: int = 30):
        """
        Ritorna (mult_q15, rshift) tali che  accum_i32 * (Sx*Sw)/S_pre ≈ (accum_i32 * mult_q15) >> rshift
        Robusto a S_pre ~ 0 e senza loop infinito.
        """
        eps = 1e-12
        Sx = float(Sx)
        Sw = float(Sw)
        S_pre = float(max(S_pre, eps))
        M = (Sx * Sw) / S_pre
        # casi degeneri
        if not math.isfinite(M) or M <= eps:
            return 0, 0
        # M = mant * 2**exp, con mant in [0.5, 1)
        mant, exp = math.frexp(M)
        m = int(round(mant * (1 << 15)))  # Q15
        r = 15 - exp                      # così che M ≈ m / 2**r
        if m <= 0:
            return 0, 0
        # normalizza nei limiti (m ≤ 32767, r ≥ 0)
        while m > 32767:
            m = (m + 1) >> 1
            r -= 1
        r = int(max(0, min(max_shift, r)))
        return m, r

