import torch
import torch.nn as nn
import torch.nn.functional as F
from RandomWalk02 import *
import torch.nn.init as init

def random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=20, walk_len=10, src_node=False):
    walk_path_all = []
    for node in range(len(adj_mat)):  # each node
        walk_path = []  # length: walk_len*walk_num
        for walk in range(walk_num):  # each walk
            walk_path.append(
                node2vecWalk_3D_Traffic2(adj_mat, node, walk_len, features_tensor, 20, P=P,
                                         Graph_level=True, lower=1, upper=1, Feature_similarity=False, src_node=src_node))
        walk_path_all.append(walk_path)
    return walk_path_all

def random_walk_path2(adj_mat, features_tensor, P=0.3, walk_num=20, walk_len=10, src_node=False):
    walk_path_all = []
    for node in range(len(adj_mat)):  # each node
        walk_path = []  # length: walk_len*walk_num
        for walk in range(walk_num):  # each walk
            walk_path.append(
                node2vecWalk_3D_Traffic3(adj_mat, node, walk_len, features_tensor, 20, P=P,
                                         Graph_level=True, lower=1, upper=1, Feature_similarity=False, src_node=src_node))
        walk_path_all.append(walk_path)
    return walk_path_all

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


class NeighborAggregator(torch.nn.Module):
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

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="concat"):
        super(GCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == "concat":
            hidden = torch.cat((self_hidden, neighbor_hidden), dim=0)
            # hidden = torch.cat((hidden, torch.tensor([torch.mean(neighbor_node_values)])), dim=0)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

class GCN2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="concat"):
        super(GCN2, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, neighbor_node_features):
        hidden = self.aggregator(neighbor_node_features)

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden

class GCN3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="concat"):
        super(GCN3, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight)

    def forward(self, neighbor_node_features):
        hidden = self.aggregator(neighbor_node_features)

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden


class Lin(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Lin, self).__init__()
        self.lin1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.lin1(x))
        output = self.lin2(output)
        return torch.exp(output)

class Lin2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Lin2, self).__init__()
        self.lin1 = nn.Linear(in_features=input_dim, out_features=output_dim, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.lin1(x))
        return output

class Transformer(nn.Module):
    def __init__(self, d_model=20, nhead=5, output_dim=1, batch_first=True):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.lin1 = nn.Linear(in_features=d_model*12+18, out_features=output_dim, bias=True)

    def forward(self, src, src_node_feature):
        out = self.transformer_encoder(src)
        out = out.reshape(src.shape[0],-1)
        out = torch.cat((out, src_node_feature), dim=-1)
        out = self.lin1(out)
        return torch.exp(out)



class GCN4(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        """：L*X*\theta
        Args:
        ----------
        input_dim: int
        output_dim: int
        use_bias : bool, optional
        """
        super(GCN4, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight_space = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight_time = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight_src = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias_time = nn.Parameter(torch.Tensor(output_dim))
            self.bias_space = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight_space, a=0.1, b=0.2)
        init.uniform_(self.weight_time, a=0.1, b=0.2)
        init.uniform_(self.weight_src, a=0.1, b=0.2)
        if self.use_bias:
            init.zeros_(self.bias_time)
            init.zeros_(self.bias_space)

    def forward(self, input_space, input_time, src_feature):
        """
        Args:
        -------
        input_space: neig_numbers*input_dim
        input_time: 1*input_dim
        src_feature: 1*input_dim
        """
        space = torch.mm(input_space, self.weight_space)
        space = space.sum(dim=0)
        time = torch.mm(input_time, self.weight_time)
        time = time.sum(dim=0)
        src = torch.mm(src_feature, self.weight_src)
        if self.use_bias:
            space += self.bias_space
            time += self.bias_time
        return space+time+src

class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, attn_dim=37, use_cos_attn=True, use_bias=False):
        """:
        Args:
        ----------
        input_dim: int
        output_dim: int
        use_bias : bool, optional
        """
        super(GAT, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.use_cos_attn = use_cos_attn
        self.attn_dim=attn_dim
        self.weight_space = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight_time = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.weight_src = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.attn1 = nn.Linear(self.attn_dim * 2, 5, bias=False)
        self.attn2 = nn.Linear(5, 1, bias=False)
        if self.use_bias:
            self.bias_time = nn.Parameter(torch.Tensor(output_dim))
            self.bias_space = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight_space, a=0.1, b=0.2)
        init.uniform_(self.weight_time, a=0.1, b=0.2)
        init.uniform_(self.weight_src, a=0.1, b=0.2)
        if self.use_bias:
            init.zeros_(self.bias_time)
            init.zeros_(self.bias_space)

    def forward(self, input_space, input_time, src_feature, src_attention, space_attention, time_attention):
        """
        Args:
        -------
        input_space: neig_numbers*input_dim
        input_time: 1*input_dim
        src_feature: 1*input_dim
        src_attention: (input_dim*2) dim=1
        input_attention: neig_numbers*(input_dim*2) dim=2
        """
        if self.use_cos_attn:
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity_space = torch.nn.functional.softmax(cos(src_attention, space_attention))
            similarity_time = torch.nn.functional.softmax(cos(src_attention, time_attention))
            similarity_space = torch.unsqueeze(similarity_space,0)
            similarity_time = torch.unsqueeze(similarity_time, 0)
        else:
            attn_mat_space = torch.cat((src_attention.expand(space_attention.shape[0], self.attn_dim), space_attention), 1)
            attn_mat_time = torch.cat((src_attention.expand(time_attention.shape[0], self.attn_dim), time_attention), 1)
            similarity_space = torch.nn.functional.softmax(self.attn2(self.attn1(attn_mat_space)),dim=0)
            similarity_space = torch.swapaxes(similarity_space, 0, 1)
            similarity_time = torch.nn.functional.softmax(self.attn2(self.attn1(attn_mat_time)),dim=0)
            similarity_time = torch.swapaxes(similarity_time, 0, 1)
        space = torch.mm(similarity_space, input_space)
        space = torch.mm(space, self.weight_space)
        space = space.mean(dim=0)
        time = torch.mm(similarity_time, input_time)
        time = torch.mm(time, self.weight_time)
        time = time.mean(dim=0)
        src = torch.mm(src_feature, self.weight_src)
        if self.use_bias:
            space += self.bias_space
            time += self.bias_time
        return space+time+src
