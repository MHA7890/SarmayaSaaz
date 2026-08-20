import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True, dropout=0.2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer_encoder(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=128):
        super(NBeatsBlock, self).__init__()
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
    def __init__(self, seq_length, num_features):
        super(NBeatsModel, self).__init__()
        self.input_size = seq_length * num_features
        self.block1 = NBeatsBlock(self.input_size, self.input_size + 1)
        self.block2 = NBeatsBlock(self.input_size, self.input_size + 1)
        self.block3 = NBeatsBlock(self.input_size, self.input_size + 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1) # Flatten
        
        theta1 = self.block1(x)
        backcast1, forecast4 = theta1[:, :-1], theta1[:, -1:]
        x = x - backcast1
        
        theta2 = self.block2(x)
        backcast2, forecast5 = theta2[:, :-1], theta2[:, -1:]
        x = x - backcast2
        
        theta3 = self.block3(x)
        forecast6 = theta3[:, -1:]
        
        return forecast4 + forecast5 + forecast6

class GatedResidualNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu = nn.ELU()
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.fc2(h)
        g = self.sigmoid(self.gate(x))
        out = self.norm(x + g * h)
        return out

class TFTModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=4):
        super(TFTModel, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.grn = GatedResidualNetwork(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.grn(x)
        attn_out, _ = self.attention(x, x, x)
        out = attn_out[:, -1, :]
        return self.fc(out)
