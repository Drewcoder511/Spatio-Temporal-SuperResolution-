import torch
import torch.nn as nn

class Window(nn.Module):
    def __init__(self, window_size, hidden_dim1, hidden_dim2, output_dim):
        """邻居聚合方式实现 Arguments: ----------
        input_dim {int} -- 输入特征的维度
        output_dim {int} -- 输出特征的维度"""
        super(Window, self).__init__()\
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lin1 = nn.Linear(in_features=input_dim, out_features=hidden_dim1, bias=True)
        self.lin2 = nn.Linear(in_features=hidden_dim1, out_features=hidden_dim2, bias=True)
        self.lin3 = nn.Linear(in_features=hidden_dim2, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()

        torch.nn.init.kaiming_uniform_(self.lin1.weight)
        torch.nn.init.kaiming_uniform_(self.lin2.weight)
        torch.nn.init.kaiming_uniform_(self.lin3.weight)

    def forward(self, neighbor_feature):
        output = self.relu(self.lin1(neighbor_feature))
        output = self.relu(self.lin2(output))
        output = self.lin3(output)
        output = torch.exp(output)
        return output