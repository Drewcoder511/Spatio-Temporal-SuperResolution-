'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)
With random walk

Comparison: EXP25 (No random walk)

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
'''
time_label = torch.unsqueeze(torch.unsqueeze(torch.arange(len(features_tensor)),dim=1),dim=1)
time_label = time_label.expand(-1,features_tensor.shape[1],1)
features_tensor = torch.cat((time_label, features_tensor), 2)

walk_num = 15
walk_len = 12
walk_path_all = random_walk_path2(adj_mat, features_tensor, P=0, walk_num=walk_num, walk_len=walk_len)
walk_path_all1 = random_walk_path2(adj_mat, features_tensor, P=0, walk_num=walk_num, walk_len=walk_len)

for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

walk = []
for node_walk in walk_path_all:
    for one_walk_path in node_walk:
        sentence = [str(word) for word in one_walk_path]
        walk.append(sentence)
# build node embedding model
model = Word2Vec(walk, vector_size=18, min_count=1)
wv = model.wv # word vector
del model

# time_node: [node [two_side [time [node]](<0), [time [node]](>0)]]
time_node = []
for node in range(49):
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
                    time_node[node][1][j[0]] = [j[1]]

hidden_size = 4
batch_size = 20
Encoder1 = GAT(20, 20, attn_dim=20, use_cos_attn = True, use_bias=False)
Encoder2 = GCN3(20, 6)
Decoder1 = Lin(40, 5, 1)
Decoder2 = Lin(6*6, 10, 1)
Decoder3 = Lin2(2,1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder1.parameters())+ list(Decoder1.parameters()) + list(Decoder3.parameters()) +
                             list(Encoder2.parameters())+ list(Decoder2.parameters()), lr=0.03, weight_decay=0.0005)

for epoch in range(50):
    Encoder1.train()
    Decoder1.train()
    Encoder2.train()
    Decoder2.train()
    Decoder3.train()

    Loss = 0
    N = 0
    node = 0
    for i in range(20, 1000, 20):
        for node in range(49):
            loss = 0
            for batch in range(20):  # each time frame
                input_time = torch.zeros(len(time_node[node][0][15])+len(time_node[node][0][17]), 20)
                input_space = torch.zeros(len(time_node[node][0][19]), 20)
                input_linear_gcn = torch.zeros(6, 6)
                '''first half-------------------------------'''
                j=15
                input_time[0:len(time_node[node][0][15])] = features_tensor[i+batch+j-20,time_node[node][0][j],:]
                input_linear_gcn[0, :] = Encoder2(features_tensor[i+batch+j-20, time_node[node][0][15], :])
                j=17
                input_time[len(time_node[node][0][15]):len(time_node[node][0][17])+len(time_node[node][0][15]),:] = \
                    features_tensor[i+batch+j-20,time_node[node][0][j],:]
                input_linear_gcn[1, :] = Encoder2(features_tensor[i+batch+j-20, time_node[node][0][17], :])
                j=19
                input_space = features_tensor[i+batch+j-20,time_node[node][0][19],:]
                input_src = features_tensor[i+batch-1,node:node+1,:]
                input_linear_gcn[2, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][0][19], :])
                src_attention = torch.cat(
                    (torch.tensor(wv[str([j, node])]), features_tensor[i + batch + j - 20, node, :]), 0)
                time_attention1 = torch.tensor(wv[[str([15, NODE]) for NODE in time_node[node][0][15]]])
                time_attention2 = torch.tensor(wv[[str([17, NODE]) for NODE in time_node[node][0][17]]])
                time_attention = torch.cat((time_attention1, time_attention2),0)
                time_attention1 = features_tensor[i + batch + 15 - 20, time_node[node][0][15], :]
                time_attention2 = features_tensor[i + batch + 17 - 20, time_node[node][0][17], :]
                time_attention = torch.cat((time_attention, torch.cat((time_attention1, time_attention2),0)),1)
                space_attention = torch.cat((torch.tensor(wv[[str([19, NODE]) for NODE in time_node[node][0][19]]]),
                                             features_tensor[i + batch + 19 - 20, time_node[node][0][19], :]),1)
                minus_1_presentation = Encoder1(input_space, input_time, input_src, src_attention, space_attention, time_attention)

                '''second half-------------------------------'''
                input_time = torch.zeros(len(time_node[node][1][25])+len(time_node[node][1][23]), 20)
                input_space = torch.zeros(len(time_node[node][1][21]), 20)
                node_set = [x for x in range(49)]
                j = 25
                input_time[0:len(time_node[node][1][25])] = features_tensor[i+batch+j-20,time_node[node][1][j],:]
                input_linear_gcn[5, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][1][25], :])
                j = 23
                input_time[len(time_node[node][1][25]):len(time_node[node][1][25])+len(time_node[node][1][23])] = \
                    features_tensor[i+batch+j-20,time_node[node][1][j],:]
                input_linear_gcn[4, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][1][23], :])
                j = 21
                input_space = features_tensor[i + batch + j - 20, time_node[node][1][21], :]
                input_src = features_tensor[i + batch + 1, node:node + 1, :]
                input_linear_gcn[3, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][1][21], :])
                src_attention = torch.cat(
                    (torch.tensor(wv[str([21, node])]), features_tensor[i + batch + j - 20, node, :]), 0)
                time_attention1 = torch.tensor(wv[[str([25, NODE]) for NODE in time_node[node][1][25]]])
                time_attention2 = torch.tensor(wv[[str([23, NODE]) for NODE in time_node[node][1][23]]])
                time_attention = torch.cat((time_attention1, time_attention2), 0)
                time_attention1 = features_tensor[i + batch + 25 - 20, time_node[node][1][25], :]
                time_attention2 = features_tensor[i + batch + 23 - 20, time_node[node][1][23], :]
                time_attention = torch.cat((time_attention, torch.cat((time_attention1, time_attention2), 0)), 1)
                space_attention = torch.cat((torch.tensor(wv[[str([21, NODE]) for NODE in time_node[node][1][21]]]),
                                             features_tensor[i + batch + 21 - 20, time_node[node][1][21], :]),1)
                plus_1_presentation = Encoder1(input_space, input_time, input_src, src_attention, space_attention, time_attention)

                graph_present = torch.cat((minus_1_presentation,plus_1_presentation),1).squeeze()
                input_linear_gcn = input_linear_gcn.reshape(-1)

                output1 = Decoder1(graph_present)
                output2 = Decoder2(input_linear_gcn)
                output = Decoder3(torch.cat((output1, output2), 0))

                y = features_tensor[i+batch, node, -1]
                loss += (output - y).pow(2).sum()
                N += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()

    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    Encoder1.eval()
    Decoder1.eval()
    Encoder2.eval()
    Decoder2.eval()
    Decoder3.eval()
    Loss = 0

    valid_pred = torch.zeros(550, 49)  # time, node
    for i in range(1000, 1550):
        for node in range(49):
            input_time = torch.zeros(len(time_node[node][0][15]) + len(time_node[node][0][17]), 20)
            input_space = torch.zeros(len(time_node[node][0][19]), 20)
            input_linear_gcn = torch.zeros(6, 6)
            '''first half-------------------------------'''
            j = 15
            input_time[0:len(time_node[node][0][15])] = features_tensor[i + j - 20, time_node[node][0][j], :]
            input_linear_gcn[0, :] = Encoder2(features_tensor[i + j - 20, time_node[node][0][15], :])
            j = 17
            input_time[len(time_node[node][0][15]):len(time_node[node][0][17]) + len(time_node[node][0][15]), :] = \
                features_tensor[i + j - 20, time_node[node][0][j], :]
            input_linear_gcn[1, :] = Encoder2(features_tensor[i + j - 20, time_node[node][0][17], :])
            j = 19
            input_space = features_tensor[i + j - 20, time_node[node][0][19], :]
            input_src = features_tensor[i + batch - 1, node:node + 1, :]
            input_linear_gcn[2, :] = Encoder2(features_tensor[i + j - 20, time_node[node][0][19], :])
            src_attention = torch.cat(
                (torch.tensor(wv[str([j, node])]), features_tensor[i + j - 20, node, :]), 0)
            time_attention1 = torch.tensor(wv[[str([15, NODE]) for NODE in time_node[node][0][15]]])
            time_attention2 = torch.tensor(wv[[str([17, NODE]) for NODE in time_node[node][0][17]]])
            time_attention = torch.cat((time_attention1, time_attention2), 0)
            time_attention1 = features_tensor[i + 15 - 20, time_node[node][0][15], :]
            time_attention2 = features_tensor[i + 17 - 20, time_node[node][0][17], :]
            time_attention = torch.cat((time_attention, torch.cat((time_attention1, time_attention2), 0)), 1)
            space_attention = torch.cat((torch.tensor(wv[[str([19, NODE]) for NODE in time_node[node][0][19]]]),
                                         features_tensor[i + 19 - 20, time_node[node][0][19], :]),1)
            minus_1_presentation = Encoder1(input_space, input_time, input_src, src_attention, space_attention,
                                           time_attention)
            '''second half-------------------------------'''
            input_time = torch.zeros(len(time_node[node][1][25]) + len(time_node[node][1][23]), 20)
            input_space = torch.zeros(len(time_node[node][1][21]), 20)
            node_set = [x for x in range(49)]
            j = 25
            input_time[0:len(time_node[node][1][25])] = features_tensor[i + j - 20, time_node[node][1][j], :]
            input_linear_gcn[5, :] = Encoder2(features_tensor[i + batch + j - 20, time_node[node][1][25], :])
            j = 23
            input_time[len(time_node[node][1][25]):len(time_node[node][1][25]) + len(time_node[node][1][23])] = \
                features_tensor[i + j - 20, time_node[node][1][j], :]
            input_linear_gcn[4, :] = Encoder2(features_tensor[i + j - 20, time_node[node][1][23], :])
            j = 21
            input_space = features_tensor[i + j - 20, time_node[node][1][21], :]
            input_src = features_tensor[i + 1, node:node + 1, :]
            input_linear_gcn[3, :] = Encoder2(features_tensor[i + j - 20, time_node[node][1][21], :])
            src_attention = torch.cat(
                (torch.tensor(wv[str([21, node])]), features_tensor[i + j - 20, node, :]), 0)
            time_attention1 = torch.tensor(wv[[str([25, NODE]) for NODE in time_node[node][1][25]]])
            time_attention2 = torch.tensor(wv[[str([23, NODE]) for NODE in time_node[node][1][23]]])
            time_attention = torch.cat((time_attention1, time_attention2), 0)
            time_attention1 = features_tensor[i + 25 - 20, time_node[node][1][25], :]
            time_attention2 = features_tensor[i + 23 - 20, time_node[node][1][23], :]
            time_attention = torch.cat((time_attention, torch.cat((time_attention1, time_attention2), 0)), 1)
            space_attention = torch.cat((torch.tensor(wv[[str([21, NODE]) for NODE in time_node[node][1][21]]]),
                                         features_tensor[i + 21 - 20, time_node[node][1][21], :]),1)
            plus_1_presentation = Encoder1(input_space, input_time, input_src, src_attention, space_attention,
                                          time_attention)

            graph_present = torch.cat((minus_1_presentation, plus_1_presentation), 1).squeeze()
            input_linear_gcn = input_linear_gcn.reshape(-1)

            output1 = Decoder1(graph_present)
            output2 = Decoder2(input_linear_gcn)
            output = Decoder3(torch.cat((output1, output2), 0))

            y = features_tensor[i, node, -1]
            loss += (output - y).pow(2).sum()
            N += 1

        valid_pred[i-1000, node] = output.item()
    Loss = (valid_pred - features_tensor[1000:1550, :, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP28_best_pred"+str(epoch))