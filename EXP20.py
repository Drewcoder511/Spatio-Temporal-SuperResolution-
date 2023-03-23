'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)

Single GCN to capture the whole graph (Time and space)
Encoder: SageGCN: 19 to 19
Decoder: Linear(38,1) Two graph representation from node 0(t=21) and node 0 (t=19)
Validation loss:
'''

import numpy as np
import torch
import copy

from BiRNN import *
from gensim.models import Word2Vec
torch.set_default_tensor_type(torch.DoubleTensor)
#torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()

features_tensor = torch.from_numpy(features)

for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

walk_num = 15
walk_len = 12
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=walk_num, walk_len=walk_len)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=walk_num, walk_len=walk_len)

for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

# time_node: [node [two_side [time [node]](<0), [time [node]](>0)]]
time_node = []
for node in range(49):
    time_node.append([{},{}])
    for i in sorted(walk_path_all[node]):
        for j in i: # i: one walk path
            if j[0] < 20:
                if j[0] in time_node[node][0]:
                    if j[1] not in time_node[node][0][j[0]]:
                        time_node[node][0][j[0]].append(j[1])
                else:
                    time_node[node][0][j[0]] = [j[1]]
            else:
                if j[0] in time_node[node][1]:
                    if j[1] not in time_node[node][1][j[0]]:
                        time_node[node][1][j[0]].append(j[1])
                else:
                    time_node[node][1][j[0]] = [j[1]]

hidden_size = 4
batch_size = 20
Encoder = GCN4(19, 19, use_bias=False)
Decoder = Lin(38, 5, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(50):
    Encoder.train()
    Decoder.train()
    Loss = 0
    N = 0
    node = 0
    for i in range(20, 1000, 20):
        for node in range(1):
            loss = 0
            for batch in range(20):  # each time frame
                '''first half-------------------------------'''
                input_time_previous_index = {} #key: node, value: index
                input_time_current_index = {}
                for j in sorted(time_node[node][0]): # first half
                    input_time_current = torch.zeros(len(time_node[node][0][j]),19)
                    for index, k in enumerate(time_node[node][0][j]): # j is a list of node set
                        input_time_current_index[k] = index
                        if k in input_time_previous_index:
                            input_time = input_time_previous[index:index+1]
                        else:
                            input_time = torch.zeros(1,19)
                        node_set = copy.deepcopy(time_node[node][0][j])
                        node_set.remove(k)
                        input_space = features_tensor[i+batch+j-20,node_set,:]
                        # 先看对应节点是否有上一个时间的input_time
                        input_src = features_tensor[i+batch+j-20,k:k+1,:]
                        input_time_current[index,:] = Encoder(input_space, input_time, input_src)
                    input_time_previous_index = copy.deepcopy(input_time_current_index)
                    input_time_current_index = {}
                    # norm:
                    """
                    #print("before",input_time_current)
                    for k in range(19):
                        input_time_current[:, k] = (input_time_current[:, k] - torch.mean(input_time_current[:, k]))
                        input_time_current[:, k] = input_time_current[:, k] / torch.max(torch.abs(input_time_current[:, k]))
                    """
                    input_time_previous = input_time_current
                    #print("after",input_time_previous)
                minus_1_presentation = input_time_previous[node]

                '''second half-------------------------------'''
                input_time_previous_index = {}  # key: node, value: index
                input_time_current_index = {}
                input_time_previous = torch.zeros(1, 19)
                for j in sorted(time_node[node][1], reverse=True):
                    input_time_current = torch.zeros(len(time_node[node][1][j]), 19)
                    for index, k in enumerate(time_node[node][1][j]):
                        if k in input_time_previous_index:
                            input_time = input_time_previous[index:index+1]
                        else:
                            input_time = torch.zeros(1,19)
                        node_set = copy.deepcopy(time_node[node][1][j])
                        node_set.remove(k)
                        input_space = features_tensor[i+batch+j-20,node_set,:]
                        input_src = features_tensor[i+batch+j-20,k:k+1,:]
                        input_time_current[index,:] = Encoder(input_space, input_time, input_src)
                    input_time_previous_index = copy.deepcopy(input_time_current_index)
                    input_time_current_index = {}
                    """
                    # norm:
                    for k in range(19):
                        input_time_current[:, k] = (input_time_current[:, k] - torch.mean(input_time_current[:, k]))
                        input_time_current[:, k] = input_time_current[:, k] / torch.max(torch.abs(input_time_current[:, k]))
                    #print(input_time_current)"""
                    input_time_previous = input_time_current
                plus_1_presentation = input_time_previous[node]
                #print("+-:",minus_1_presentation,plus_1_presentation)
                output = Decoder(torch.cat((minus_1_presentation,plus_1_presentation),0))
                #print("output:",output)

                y = features_tensor[i+batch, node, -1]
                loss += (output - y).pow(2).sum()

                N += 1
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()

    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    Encoder.eval()
    Decoder.eval()
    Loss = 0

    valid_pred = torch.zeros(550, 1)  # time, node
    for i in range(1000, 1550):
        node = 0

        input_time_previous_index = {}  # key: node, value: index
        input_time_current_index = {}
        input_time_previous = torch.zeros(1, 19)
        for j in sorted(time_node[node][0]):  # first half
            input_time_current = torch.zeros(len(time_node[node][0][j]), 19)
            for index, k in enumerate(time_node[node][0][j]):  # j is a list of node set
                if k in input_time_previous:
                    input_time = input_time_previous[index:index+1]
                else:
                    input_time = torch.zeros(1,19)
                node_set = copy.deepcopy(time_node[node][0][j])
                node_set.remove(k)
                input_space = features_tensor[i + j - 20, node_set, :]
                # 先看对应节点是否有上一个时间的input_time
                input_src = features_tensor[i + j - 20, k:k+1, :]
                input_time_current[index,:] = Encoder(input_space, input_time, input_src)
                # norm:
            """for k in range(19):
                mean = torch.mean(input_time_current[:, k])
                std = torch.std(input_time_current[:, k], unbiased=False)
                input_time_current[:, k] = (input_time_current[:, k] - mean) / std"""
            input_time_previous = input_time_current
        minus_1_presentation = input_time_previous[node]

        input_time_previous_index = {}  # key: node, value: index
        input_time_current_index = {}
        input_time_previous = torch.zeros(1, 19)
        for j in sorted(time_node[node][1], reverse=True):  # second half
            input_time_current = torch.zeros(len(time_node[node][1][j]), 19)
            for index, k in enumerate(time_node[node][1][j]):
                if k in input_time_previous:
                    input_time = input_time_previous[index:index+1]
                else:
                    input_time = torch.zeros(1, 19)
                node_set = copy.deepcopy(time_node[node][1][j])
                node_set.remove(k)
                input_space = features_tensor[i + j - 20, node_set, :]
                input_src = features_tensor[i + j - 20, k:k+1, :]
                # 先看对应节点是否有上一个时间的input_time
                input_time_current[index,:] = Encoder(input_space, input_time, input_src)
            # norm:
            """for k in range(19):
                mean = torch.mean(input_time_current[:, k])
                max = torch.max(torch.abs(input_time_current[:, k]))
                input_time_current[:, k] = (input_time_current[:, k] - mean) / max"""
            input_time_previous = input_time_current
        plus_1_presentation = input_time_previous[node]

        output = Decoder(torch.cat((minus_1_presentation, plus_1_presentation),0))

        y = features_tensor[i, node, -1]
        loss += (output - y).pow(2).sum()
        N += 1

        valid_pred[i-1000, node] = output.item()
    Loss = (valid_pred - features_tensor[1000:1550, node, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP17_best_pred"+str(epoch))