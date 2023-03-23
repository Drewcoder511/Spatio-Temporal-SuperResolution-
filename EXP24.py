'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)

Single GCN to capture the whole graph (Time and space)
Encoder: Three individual SageGCN: 19 to 19
Decoder: Linear(38,1) Two graph representation from node 0(t=21) and node 0 (t=19)
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


hidden_size = 4
batch_size = 20
Encoder1 = GAT(20, 20, attn_dim=20, use_bias=False)
Encoder2 = GAT(20, 20, attn_dim=20, use_bias=False)
Encoder3 = GAT(20, 20, attn_dim=20, use_bias=False)
Decoder = Lin(40, 5, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder1.parameters()) + list(Encoder2.parameters()) +
                             list(Encoder3.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(50):
    Encoder1.train()
    Encoder2.train()
    Encoder3.train()
    Decoder.train()

    Loss = 0
    N = 0
    node = 0
    for i in range(20, 1000, 20):
        for node in range(1):
            loss = 0
            for batch in range(20):  # each time frame
                input_time_previous = torch.zeros(49,20)
                '''first half-------------------------------'''
                for j in [15,17,19]: # first half
                    input_time_current = torch.zeros(49,20)
                    if j==15:
                        input_time = torch.zeros(1,20)
                    else:
                        input_time = input_time_previous

                    for index in range(49): # j is time
                        node_set = [x for x in range(49)]
                        node_set.remove(index)
                        input_space = features_tensor[i+batch+j-20,node_set,:] ##########改
                        # 先看对应节点是否有上一个时间的input_time
                        input_src = features_tensor[i+batch+j-20, index:index+1, :]
                        src_attention = features_tensor[i+batch+j-20, index, :]
                        input_attention = features_tensor[i+batch+j-20,node_set,:]
                        if j == 15:
                            input_time_current[index, :] = Encoder1(input_space, input_time, input_src, src_attention, input_attention)
                        elif j == 17:
                            input_time_current[index, :] = Encoder2(input_space, input_time, input_src, src_attention, input_attention)
                        elif j == 19:
                            input_time_current[index, :] = Encoder3(input_space, input_time, input_src, src_attention, input_attention)
                    input_time_previous = input_time_current
                minus_1_presentation = input_time_previous[node]

                '''second half-------------------------------'''
                for j in [25, 23, 21]:  # second half
                        input_time_current = torch.zeros(49, 20)
                        if j == 25:
                            input_time = torch.zeros(1, 20)
                        else:
                            input_time = input_time_previous

                        for index in range(49):  # j is time
                            node_set = [x for x in range(49)]
                            node_set.remove(index)
                            input_space = features_tensor[i + batch + j - 20, node_set, :]  ##########改
                            # 先看对应节点是否有上一个时间的input_time
                            input_src = features_tensor[i + batch + j - 20, index:index + 1, :]
                            src_attention = features_tensor[i + batch + j - 20, index, :]
                            input_attention = features_tensor[i + batch + j - 20, node_set, :]
                            if j == 25:
                                input_time_current[index, :] = Encoder1(input_space, input_time, input_src,
                                                                        src_attention, input_attention)
                            elif j == 23:
                                input_time_current[index, :] = Encoder2(input_space, input_time, input_src,
                                                                        src_attention, input_attention)
                            elif j == 21:
                                input_time_current[index, :] = Encoder3(input_space, input_time, input_src,
                                                                        src_attention, input_attention)
                        input_time_previous = input_time_current
                plus_1_presentation = input_time_previous[node]
                output = Decoder(torch.cat((minus_1_presentation,plus_1_presentation),0))
                y = features_tensor[i+batch, node, -1]
                loss += (output - y).pow(2).sum()
                N += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()

    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    Encoder1.eval()
    Encoder2.eval()
    Encoder3.eval()
    Decoder.eval()
    Loss = 0

    valid_pred = torch.zeros(550, 1)  # time, node
    for i in range(1000, 1550):
        node = 0
        input_time_previous = torch.zeros(1, 20)
        for j in [15, 17, 19]:  # first half
            input_time_current = torch.zeros(49, 20)
            if j == 15:
                input_time = torch.zeros(1, 20)
            else:
                input_time = input_time_previous

            for index in range(49):  # j is time
                node_set = [x for x in range(49)]
                node_set.remove(index)
                input_space = features_tensor[i + j - 20, node_set, :]  ##########改
                # 先看对应节点是否有上一个时间的input_time
                input_src = features_tensor[i + j - 20, index:index + 1, :]
                src_attention = features_tensor[i + j - 20, index, :]
                input_attention = features_tensor[i + j - 20, node_set, :]
                if j == 15:
                    input_time_current[index, :] = Encoder1(input_space, input_time, input_src, src_attention,
                                                            input_attention)
                elif j == 17:
                    input_time_current[index, :] = Encoder2(input_space, input_time, input_src, src_attention,
                                                            input_attention)
                elif j == 19:
                    input_time_current[index, :] = Encoder3(input_space, input_time, input_src, src_attention,
                                                            input_attention)
            input_time_previous = input_time_current
        minus_1_presentation = input_time_previous[node]

        input_time_previous = torch.zeros(1, 20)
        for j in [25, 23, 21]:  # second half
            input_time_current = torch.zeros(49, 20)
            if j == 25:
                input_time = torch.zeros(1, 20)
            else:
                input_time = input_time_previous

            for index in range(49):  # j is time
                node_set = [x for x in range(49)]
                node_set.remove(index)
                input_space = features_tensor[i + j - 20, node_set, :]  ##########改
                # 先看对应节点是否有上一个时间的input_time
                input_src = features_tensor[i + j - 20, index:index + 1, :]
                src_attention = features_tensor[i + j - 20, index, :]
                input_attention = features_tensor[i + j - 20, node_set, :]
                if j == 25:
                    input_time_current[index, :] = Encoder1(input_space, input_time, input_src,
                                                            src_attention, input_attention)
                elif j == 23:
                    input_time_current[index, :] = Encoder2(input_space, input_time, input_src,
                                                            src_attention, input_attention)
                elif j == 21:
                    input_time_current[index, :] = Encoder3(input_space, input_time, input_src,
                                                            src_attention, input_attention)
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
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP24_best_pred"+str(epoch))