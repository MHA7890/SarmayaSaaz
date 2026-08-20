"""
PyTorch architecture definitions.

These are the single source of truth for every neural net in the system. The
class bodies must stay byte-compatible with the code that trained them, because
state_dict loading matches on parameter names and shapes.

Previously these were redeclared in four places (src/commodities/inference_utils.py,
src/crypto/stage4_master_training.py, src/mufap/stage5_inference.py,
src/stocks/stage5_inference.py) - MUFAP_LSTM and PSX_LSTM were identical classes
under different names. Consolidated here.
"""
import torch
import torch.nn as nn

# ============================================================================
# Commodities  (hidden_size=32, seq_length=10, 1 layer)
# ============================================================================


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class GRUModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=0.2
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.transformer_encoder(self.input_proj(x))
        return self.fc(self.dropout(out[:, -1, :]))


class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, theta_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        return self.fc3(h)


class NBeatsModel(nn.Module):
    """Doubly-residual stack: each block subtracts its backcast, forecasts sum."""

    def __init__(self, seq_length: int, num_features: int):
        super().__init__()
        self.input_size = seq_length * num_features
        self.block1 = NBeatsBlock(self.input_size, self.input_size + 1)
        self.block2 = NBeatsBlock(self.input_size, self.input_size + 1)
        self.block3 = NBeatsBlock(self.input_size, self.input_size + 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        t1 = self.block1(x)
        x = x - t1[:, :-1]
        t2 = self.block2(x)
        x = x - t2[:, :-1]
        t3 = self.block3(x)
        return t1[:, -1:] + t2[:, -1:] + t3[:, -1:]


class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        h = self.fc2(self.elu(self.fc1(x)))
        return self.norm(x + self.sigmoid(self.gate(x)) * h)


class TFTModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.grn = GatedResidualNetwork(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.grn(self.input_proj(x))
        attn_out, _ = self.attention(x, x, x)
        return self.fc(attn_out[:, -1, :])


# ============================================================================
# Crypto  (hidden=64, seq_length=30)
# ============================================================================


class CryptoLSTM(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.lstm = nn.LSTM(f, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CryptoGRU(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.gru = nn.GRU(f, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class CryptoTransformer(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.emb = nn.Linear(f, 64)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True),
            num_layers=2,
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.encoder(self.emb(x)).mean(dim=1))


class NBEATS_Lite(nn.Module):
    def __init__(self, f: int, seq_len: int = 30):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(f * seq_len, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.fc(x.reshape(x.size(0), -1))


class TFT_Lite(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.lstm = nn.LSTM(f, 64, batch_first=True)
        self.gate = nn.Linear(64, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last * torch.sigmoid(self.gate(last)))


# ============================================================================
# MUFAP + PSX  (tabular LSTM: single timestep, hidden=64, 2 layers)
# ============================================================================


class TabularLSTM(nn.Module):
    """
    Consumes a single feature row, not a sequence - unsqueeze(1) creates a
    length-1 time dimension. Used identically by the MUFAP and PSX engines
    (formerly duplicated as MUFAP_LSTM and PSX_LSTM).
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.2,
        )
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        out = self.dropout(self.relu(self.fc1(out[:, -1, :])))
        return self.fc2(out).squeeze(1)


# ============================================================================
# Introspection
# ============================================================================

def infer_input_width(state_dict: dict, seq_length: int) -> int | None:
    """
    Recover the feature width a serialized model was trained on, by reading the
    shape of its first weight tensor.

    This is what makes the champion/challenger revert survivable: 154 commodity
    artifacts expect 9 fewer features (the sentiment block) than their scaler
    emits, and the width varies per commodity (25/26/27/28). Detecting it beats
    hardcoding one commodity's number.
    """
    for key in ("lstm.weight_ih_l0", "gru.weight_ih_l0"):
        if key in state_dict:
            return state_dict[key].shape[1]
    for key in ("input_proj.weight", "emb.weight"):
        if key in state_dict:
            return state_dict[key].shape[1]
    # N-BEATS flattens seq x features into the first Linear.
    #
    # Two key layouts exist for the same network. The original pipeline named
    # its three blocks as separate attributes (block1/block2/block3); the
    # training-scripts-new pipeline holds them in an nn.ModuleList, which
    # serializes as blocks.0/blocks.1/blocks.2. The maths and every tensor
    # shape are identical - verified max|diff| == 0.0 between the two - so both
    # layouts are accepted here and normalized in registry._load_torch.
    for key in ("block1.fc1.weight", "blocks.0.fc1.weight", "fc.0.weight"):
        if key in state_dict:
            return state_dict[key].shape[1] // seq_length
    return None
