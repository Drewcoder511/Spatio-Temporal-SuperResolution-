import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, dropout=0.4, training=False):
        #怎么把时间的标签放进feature里，让他和别的feature不一样
        #怎么把random walk路径变成一个子图，怎么构建这么多个子图，49个
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=training)
        x = self.conv2(x, edge_index)
        x = torch.exp(x)
        return x