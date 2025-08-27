"""
Neural network definitions.
This version of **LSTMModel** adds a simple dot-product attention
mechanism over the temporal dimension.

The goal of our task is to predict the junction temperature at the
*current* timestep given a fixed-length sliding window of input signals
(power and base-plate temperature). A plain LSTM already condenses the
whole sequence into its last hidden state, but attention often helps by
letting the model selectively focus on the most relevant timesteps in
the window.

Below we keep the same public interface - the model still accepts a
tensor of shape *(B, W, F)* and outputs a vector of shape *(B,)* - so
no change is required anywhere else in the training / evaluation code.

Implementation details
----------------------
1. **Encoder** A single-layer LSTM returns the full sequence of hidden
   states `H ∈ ℝ^{B×W×H}` and the final hidden state `h_T ∈ ℝ^{B×H}`.
2. **Attention scores** For each timestep *t* we compute a dot product
   `score_t = h_t · h_T`.  In matrix form this is a *batched* matrix
   multiplication handled efficiently by `torch.bmm`.
3. **Weights** We normalise the scores with a softmax over the window
   dimension to obtain attention weights `α_t` that sum to 1.
4. **Context vector** A weighted sum of the encoder states produces a
   context vector `c ∈ ℝ^{B×H}`.
5. **Prediction head** Finally we concatenate the context vector and
   the query (last hidden state) and feed the result to a small linear
   layer that outputs the predicted temperature.

This is a *very* lightweight mechanism (O(W·H) extra operations) and is
well suited for small hidden sizes like the 16 units used in `config.py`.
"""

from __future__ import annotations  # for future-proof type hints

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """Single-layer LSTM + dot-product attention (sequence → scalar)."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.10):
        """Create the encoder and the tiny fully-connected prediction head.

        Parameters
        ----------
        input_size : int
            Number of features per timestep (2 in our dataset: *P*, *Tbp*).
        hidden_size : int
            Size of the LSTM hidden state vector.
        num_layers : int, optional
            Number of stacked LSTM layers (default: 1).  More layers would
            require only minor tweaks to the attention (we still use the last
            layer's hidden state as query).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p   = dropout_p

        # 1) Recurrent encoder
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # 2) Prediction head - takes [query ; context] → scalar temperature
        self.fc   = nn.Linear(hidden_size * 2, 1)

    # ---------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, *, mc_dropout: bool = False) -> torch.Tensor:  # x shape: (B, W, F)
        """Run the sequence through the LSTM and apply temporal attention.

        Returns
        -------
        torch.Tensor
            Predicted junction temperature - shape *(B,)*.
        """
        # ------------------------------------------------ encoder
        # `outputs` contains the hidden state at **every** timestep, while
        # `h_T` is the hidden state of the **last** timestep (for all layers).
        outputs, (h_T, _) = self.lstm(x)  # outputs: (B, W, H); h_T: (L, B, H)

        # We take the last layer's hidden state as the *query* vector.
        query = h_T[-1]  # shape: (B, H)

        # ------------------------------------------------ attention scores
        # Compute unnormalised attention scores via batched dot product.
        #   scores[b, t] = outputs[b, t] · query[b]
        scores = torch.bmm(outputs, query.unsqueeze(2)).squeeze(2)  # (B, W)

        # Softmax over the **window** dimension so weights sum to 1.
        weights = F.softmax(scores, dim=1)  # (B, W)

        # ------------------------------------------------ context vector
        # Weighted sum: context[b] = Σ_t α_t · h_t
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (B, H)

        # ------------------------------------------------ prediction head
        combined = torch.cat([query, context], dim=1)  # (B, 2H)
        combined = F.dropout(combined, p=self.dropout_p, training=mc_dropout)
        out = self.fc(combined).squeeze(-1)            # (B,)
        return out
