import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def NeigAgg_3D(adj_mat, start_node, sample_nums, data, t, P=0.3, Graph_level=True, lower=1, upper=1,
               Feature_similarity=False):  # include node features

    # edge_index:
    # start_node: the node to start random walk

    # sample_nums:  {list of int} -- 每一阶需要采样的个数

    # features: features of time t(features[t][node_num])
    # data (data[t].x, data[t].y)

    # t: time frame of the starting node
    # 删掉了p，q，因为49节点问题没有距离为0和1的节点（全联接）
    # 随时间相隔越来越远的概率衰减函数是 0.5 + 1/t

    gap = lower + upper  # The gap to skip

    if Graph_level:  # Graph level prediction so no random walk on frame t
        if t - lower < 0:  # if start_node at the first frame
            Dis_1_time = [t + upper]
        elif t + upper > len(data) - 1:  # if start_node at the last frame
            Dis_1_time = [t - lower]
        else:
            Dis_1_time = [t - lower, t + upper]

    # random choose one node as the first step
    pro = adj_mat[start_node]  # 49 nodes
    curnbr = np.arange(49)
    pro = pro / np.sum(pro)  # normalization probabilities

    # return sampling_result [list of ndarray] 每一阶采样的结果

    sampling_result = [[]]
    for nums in range(sample_nums[0]):
        if len(Dis_1_time) == 1:
            time = Dis_1_time[0]
        else:
            time = np.random.choice(np.array(Dis_1_time), p=np.array([0.5, 0.5]))  # 1/2 pb choose one time
        # 改：从49个随机选改成只移动到自己
        node = start_node
        sampling_result[0].append([time, node])  ### Dis_1变1:n的数组然后再索引回来

    for k, hopk_num in enumerate(sample_nums[1:]):
        sampling_result.append([])
        for sample in sampling_result[k]:
            current_t = sample[0]
            current_node = sample[1]
            for num in range(hopk_num):
                Nbr_with_t = [current_t - gap, current_t, current_t + gap]
                pro_t = np.zeros(3)
                # jump low
                pro_t[0] = 1 / abs(Nbr_with_t[0] - t)
                # stay
                pro_t[1] = 1 / abs(Nbr_with_t[1] - t) + P  # more perfer to stay at current time
                # jump high
                pro_t[2] = 1 / abs(Nbr_with_t[2] - t)
                pro_t = pro_t / np.sum(pro_t)  # norm
                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)

                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node
                sampling_result[k + 1].append([time, node])
    return sampling_result

def walk_to_feature(walk_result, t, features_tensor):
    walk_feature = []
    for i in walk_result:
        khop_feature = torch.zeros(len(i), 19)
        for index, j in enumerate(i):
            khop_feature[index] = features_tensor[j[0]-20+t,j[1],:]
        walk_feature.append(khop_feature)
    return walk_feature


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
        # print(aggr_neighbor.shape)
        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        return neighbor_hidden


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim_neig, hidden_dim_src, activation=F.relu,
                 aggr_neighbor_method="mean", aggr_hidden_method="concat"):
        super(GCN, self).__init__()
        assert aggr_neighbor_method in ["mean", "sum", "max"]
        assert aggr_hidden_method in ["sum", "concat"]
        self.aggr_neighbor = aggr_neighbor_method
        self.aggr_hidden = aggr_hidden_method
        self.activation = activation
        self.aggregator = NeighborAggregator(input_dim, hidden_dim_neig, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim_src))
        self.hidden_dim_neig = hidden_dim_neig

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.aggregator(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)
        if self.aggr_hidden == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden == "concat":
            # self_hidden = self_hidden.mean(dim=0)
            neighbor_hidden = neighbor_hidden + torch.zeros(self_hidden.shape[0], self.hidden_dim_neig)
            # print(self_hidden.shape, neighbor_hidden.shape)
            hidden = torch.cat((self_hidden, neighbor_hidden), dim=1)
            # hidden = torch.cat((hidden, torch.tensor([torch.mean(neighbor_node_values)])), dim=0)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hidden))

        if self.activation:
            return self.activation(hidden)
        else:
            return hidden


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim_neig, hidden_dim_src, src_dim, output_dim, k_hop_num):
        super(Decoder, self).__init__()
        self.k_hop_num = k_hop_num
        self.gcn1 = GCN(input_dim, hidden_dim_neig, hidden_dim_src, aggr_neighbor_method="mean")
        self.gcn2 = GCN(input_dim, hidden_dim_neig, hidden_dim_src, aggr_neighbor_method="mean")
        self.gcn3 = GCN(input_dim, hidden_dim_neig, hidden_dim_src, aggr_neighbor_method="mean")
        self.lin1 = nn.Linear(in_features=((hidden_dim_neig + hidden_dim_src) * 4), out_features=output_dim, bias=True)
        self.lin2 = nn.Linear(in_features=src_dim, out_features=output_dim, bias=True)

    def forward(self, x, y):
        #output = self.gcn1(x[2], x[3])
        #x[2] = output
        #output = self.gcn2(x[1], x[2])
        #x[1] = output
        output = self.gcn3(x[0], x[1])
        x[0] = output

        output = output.reshape(-1)
        output = self.lin1(output)
        #print("1,", output)
        output = output + self.lin2(y)
        #print("2,", output)
        #return torch.exp(output)
        return output