"""
PyTorch model architectures, matched 1:1 to what each original pipeline
trained (see the stage4/generator source each class references). Kept in one
shared module so crypto's and commodities' near-duplicate LSTM/GRU/Transformer
definitions don't drift from each other, and so inference code (if the new
models are wired into the backend later) has one place to import from.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Routed engines (PSX stocks, MUFAP): a single feature row treated as a
# seq_len=1 "sequence" - see src/stocks/stage4_master_training.py::PSX_LSTM
# and src/mufap/stage4_master_training.py::MUFAP_LSTM (identical shape).
# ---------------------------------------------------------------------------
class TabularLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2,
                             batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, features) -> (batch, seq_len=1, features)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Crypto (src/crypto/stage4_master_training.py) - SEQ_LEN=30, hidden=64, no
# dropout, full sequence models (not seq_len=1 like the routed engines).
# ---------------------------------------------------------------------------
class CryptoLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CryptoGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class CryptoTransformer(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.emb = nn.Linear(input_dim, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc(self.encoder(self.emb(x)).mean(dim=1))  # mean-pool over sequence


class NBeatsLite(nn.Module):
    """Crypto's 'NBEATS_Lite': a single hidden layer over the flattened window,
    not a real doubly-residual N-BEATS (that's CommodityNBeats below)."""

    def __init__(self, input_dim: int, seq_len: int, hidden: int = 128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim * seq_len, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


class TFTLite(nn.Module):
    """Crypto's 'TFT_Lite': LSTM + a GLU-style self-gate, not a real TFT
    (that's CommodityTFT below)."""

    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        gated = last * torch.sigmoid(self.gate(last))
        return self.fc(gated)


# ---------------------------------------------------------------------------
# Commodities (src/generators/create_dl_nb_{h}d.py) - SEQ_LEN=10, hidden=32,
# dropout 0.25 (LSTM/GRU/NBEATS) or 0.2 (Transformer/TFT internal), a real
# 3-block doubly-residual N-BEATS, and a lite-GRN+attention TFT.
# ---------------------------------------------------------------------------
class CommodityLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class CommodityGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


class CommodityTransformer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.transformer_encoder(self.input_proj(x))
        return self.fc(self.dropout(h[:, -1, :]))


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, theta_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        return self.fc3(h)


class CommodityNBeats(nn.Module):
    """Real doubly-residual N-BEATS: 3 stacked blocks, each block's backcast
    is subtracted from the running input, forecasts summed."""

    def __init__(self, input_size: int, seq_len: int, num_blocks: int = 3, hidden: int = 128):
        super().__init__()
        flat_size = input_size * seq_len
        theta_size = flat_size + 1  # backcast (flat_size) + 1 forecast scalar
        self.flat_size = flat_size
        self.blocks = nn.ModuleList([NBeatsBlock(flat_size, theta_size, hidden) for _ in range(num_blocks)])

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        residual = x
        forecast = 0.0
        for block in self.blocks:
            theta = block(residual)
            backcast, block_forecast = theta[:, : self.flat_size], theta[:, self.flat_size :]
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast  # shape (batch, 1)


class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.fc2(self.dropout(self.elu(self.fc1(x))))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(x + gate * h)


class CommodityTFT(nn.Module):
    """Lite TFT: input projection -> GRN -> single self-attention block ->
    last timestep -> linear head. Not a full TFT (no variable-selection
    networks or quantile outputs)."""

    def __init__(self, input_size: int, hidden_size: int = 32, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.grn = GatedResidualNetwork(hidden_size, dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h = self.grn(self.input_proj(x))
        attn_out, _ = self.attention(h, h, h)
        return self.fc(attn_out[:, -1, :])
