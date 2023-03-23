'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)
With random walk

Comparison: EXP28 (Delete GAT, leaving only GCN+Linear)

Single GCN to capture the whole graph (Time and space):
-One layer
-Learning Attention

Encoder: One individual SageGCN: 20 to 20
Decoder: Linear(40,1) Two graph representation from node 0(t=21) and node 0 (t=19)
Validation loss:
'''

import numpy as np
import torch
import copy

from BiRNN import *
from gensim.models import Word2Vec
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()

for i in range(18):
    data = features[:,:,i].reshape(-1)
    features[:,:,i] = features[:,:,i] - np.mean(data)
    features[:,:,i] = features[:,:,i]/np.max(np.abs(data))

features_tensor = torch.from_numpy(features)

'''
Add time label in the data:
now the data shape is [1577,49,20]

time_label = torch.unsqueeze(torch.unsqueeze(torch.arange(len(features_tensor)),dim=1),dim=1)
time_label = time_label.expand(-1,features_tensor.shape[1],1)
features_tensor = torch.cat((time_label, features_tensor), 2)
'''
walk_num = 15
walk_len = 12

#walk_path_all = random_walk_path2(adj_mat, features_tensor, P=0, walk_num=walk_num, walk_len=walk_len)
#walk_path_all1 = random_walk_path2(adj_mat, features_tensor, P=0, walk_num=walk_num, walk_len=walk_len)
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=walk_num, walk_len=walk_len)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=walk_num, walk_len=walk_len)

for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

# time_node: [node [two_side [time [node]](<0), [time [node]](>0)]]
time_node = []
"""for node in range(49):
    time_node.append([{},{}])
    for i in sorted(walk_path_all[node]):
        for j in i: # i: one walk path
            if j[0] < 20 and j[0] > 14:
                if j[0] in time_node[node][0]:
                    if j[1] not in time_node[node][0][j[0]]:
                        time_node[node][0][j[0]].append(j[1])
                else:
                    time_node[node][0][j[0]] = [j[1]]
            elif j[0] < 26 and j[0] > 20:
                if j[0] in time_node[node][1]:
                    if j[1] not in time_node[node][1][j[0]]:
                        time_node[node][1][j[0]].append(j[1])
                else:
                    time_node[node][1][j[0]] = [j[1]]"""
time_node = []
for node in range(49):
    time_node.append({})
    for i in walk_path_all[node]:
        for j in i:
            if j[0] > 14 and j[0] < 26:
                if j[0] in time_node[node]:
                    if j[1] not in time_node[node][j[0]]:
                        time_node[node][j[0]].append(j[1])
                else:
                    time_node[node][j[0]] = [j[1]]

hidden_size = 4
batch_size = 20
Encoder = GCN2(19, 6)
Decoder = Lin(6*6, 10, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(
                             list(Encoder.parameters())+ list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(50):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    node = 0
    for i in range(20, 1000, 20):
        for node in range(49):
            loss = 0
            for batch in range(20):  # each time frame
                input_linear_gcn = torch.zeros(6, 6)
                for index, j in enumerate(sorted(time_node[node])):
                    Input = features_tensor[i+batch+j-20,time_node[node][j],:]
                    output = Encoder(Input)
                    output = output.reshape(1, 1, -1)
                    input_linear_gcn[index, :] = output
                input_linear_gcn = input_linear_gcn.reshape(-1)
                """
                '''first half-------------------------------'''
                j=15
                input_linear_gcn[0, :] = Encoder2(features_tensor[i+batch+j-20, time_node[node][15], :])
                j=17
                input_linear_gcn[1, :] = Encoder2(features_tensor[i+batch+j-20, time_node[node][17], :])
                j=19
                input_linear_gcn[2, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][19], :])

                '''second half-------------------------------'''
                j = 25
                input_linear_gcn[5, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][25], :])
                j = 23

                input_linear_gcn[4, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][23], :])
                j = 21

                input_linear_gcn[3, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][21], :])


                input_linear_gcn = input_linear_gcn.reshape(-1)"""


                output = Decoder(input_linear_gcn)


                y = features_tensor[i+batch, node, -1]
                loss += (output - y).pow(2).sum()
                N += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()

    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))


    Encoder.eval()
    Decoder.eval()

    Loss = 0

    valid_pred = torch.zeros(550, 49)  # time, node
    for i in range(1000, 1550):
        for node in range(49):

            input_linear_gcn = torch.zeros(6, 6)
            for index, j in enumerate(sorted(time_node[node])):
                Input = features_tensor[i + j - 20, time_node[node][j], :]
                output = Encoder(Input)
                output = output.reshape(1, 1, -1)
                input_linear_gcn[index, :] = output
            """
            '''first half-------------------------------'''
            j = 15

            input_linear_gcn[0, :] = Encoder2(features_tensor[i + j - 20, time_node[node][15], :])
            j = 17

            input_linear_gcn[1, :] = Encoder2(features_tensor[i + j - 20, time_node[node][17], :])
            j = 19

            input_linear_gcn[2, :] = Encoder2(features_tensor[i + j - 20, time_node[node][19], :])

            '''second half-------------------------------'''


            j = 25

            input_linear_gcn[5, :] = Encoder2(features_tensor[i  + j - 20, time_node[node][25], :])
            j = 23

            input_linear_gcn[4, :] = Encoder2(features_tensor[i + j - 20, time_node[node][23], :])
            j = 21

            input_linear_gcn[3, :] = Encoder2(features_tensor[i + j - 20, time_node[node][21], :])"""



            input_linear_gcn = input_linear_gcn.reshape(-1)


            output = Decoder(input_linear_gcn)

            y = features_tensor[i, node, -1]
            loss += (output - y).pow(2).sum()
            N += 1

        valid_pred[i-1000, node] = output.item()
    Loss = (valid_pred - features_tensor[1000:1550, :, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP29_best_pred")