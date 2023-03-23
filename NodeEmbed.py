import torch
import torch.nn as nn
import torch.nn.functional as F

class NeighborAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        """邻居聚合方式实现 Arguments: ----------
        input_dim {int} -- 输入特征的维度
        output_dim {int} -- 输出特征的维度
        Keyword Arguments: -----------------
        use_bias {bool} -- 是否使用偏置 (default: {False})
        aggr_method {string} -- 聚合方式 (default: {mean}) """
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=0)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=0)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=0)[0]
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}".format(self.aggr_method))

        '''
        neighbor_feature: 200*18
        aggr_neighbor: 1*200 -> 1*18
        self.weight: 18*10
        '''
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden


class SageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="concat"):
        super(SageGCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features, neighbor_node_values):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == "concat":
            hidden = torch.cat((self_hidden, neighbor_hidden), dim=0)
            hidden = torch.cat((hidden, torch.tensor([torch.mean(neighbor_node_values)])), dim=0)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden


class NodeEmbed(nn.Module):
    def __init__(self, input_dim, output_dim, input_size, hidden_size, output_size, aggr_neighbor_method="mean"):
        super(NodeEmbed, self).__init__()
        self.input_dim = input_dim
        self.lin1 = nn.Linear(in_features=input_size, out_features=hidden_size, bias=True)
        self.lin2 = nn.Linear(in_features=hidden_size, out_features=output_size, bias=True)
        self.relu = nn.ReLU()
        self.gcn = SageGCN(input_dim, output_dim, aggr_neighbor_method=aggr_neighbor_method)

        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.gcn.reset_parameters()

    def forward(self, src_node_features, neighbor_node_features, neighbor_node_values):
        output = self.gcn(src_node_features, neighbor_node_features, neighbor_node_values)
        output = self.relu(self.lin1(output))
        output = self.lin2(output)
        output = torch.exp(output)
        return output

    # 把时间feature加进去