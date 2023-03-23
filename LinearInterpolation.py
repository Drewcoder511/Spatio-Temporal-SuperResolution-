import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 2 for bidirection

    def forward(self, x, h0):
        # Set initial states
        # h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        # c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, h0 = self.gru(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, h0

class Transformer_LinearInt(nn.Module):
    def __init__(self, d_model=20, nhead=5, output_dim=1, batch_first=True):
        super(Transformer_LinearInt, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, batch_first=batch_first)
        self.lin1 = nn.Linear(in_features=d_model+18, out_features=output_dim, bias=True)

    def forward(self, src, tgt, src_node_feature):
        out = self.transformer(src, tgt)
        out = out.squeeze()
        out = torch.cat((out, src_node_feature), dim=-1)
        out = self.lin1(out)
        return torch.exp(out)

class LinearInt(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearInt, self).__init__()
        self.lin1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)

    def forward(self, x):
        out = F.relu(self.lin1(x))
        out = self.lin2(out)
        return out