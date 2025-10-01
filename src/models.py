from __future__ import annotations  # for future-proof type hints

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.10):
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

        outputs, (h_T, _) = self.lstm(x)  # outputs: (B, W, H); h_T: (L, B, H)
        query = h_T[-1]  # shape: (B, H)

        # ------------------------------------------------ attention scores
        scores = torch.bmm(outputs, query.unsqueeze(2)).squeeze(2)  # (B, W)
        weights = F.softmax(scores, dim=1)  # (B, W)

        # ------------------------------------------------ context vector
        context = torch.bmm(weights.unsqueeze(1), outputs).squeeze(1)  # (B, H)

        # ------------------------------------------------ prediction head
        combined = torch.cat([query, context], dim=1)  # (B, 2H)
        combined = F.dropout(combined, p=self.dropout_p, training=mc_dropout)
        out = self.fc(combined).squeeze(-1)            # (B,)
        return out
    
class LSTMModelNoA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, dropout_p: float = 0.10):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_p   = dropout_p

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, *, mc_dropout: bool = False) -> torch.Tensor:  # x: (B, W, F)

        _, (h_T, _) = self.lstm(x)     # h_T: (L, B, H)
        summary = h_T[-1]              # (B, H)
        summary = F.dropout(summary, p=self.dropout_p, training=mc_dropout)
        out = self.fc(summary).squeeze(-1)  # (B,)
        return out
