'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)

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


hidden_size = 4
batch_size = 20
Encoder = GAT(20, 20, attn_dim=20, use_cos_attn = False, use_bias=False)
Decoder = Lin(40, 5, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters())+ list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

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
                input_time = torch.zeros(49*2, 20)
                input_space = torch.zeros(48, 20)
                '''first half-------------------------------'''
                input_time_current = torch.zeros(49,20)
                node_set = [x for x in range(49)]
                j=15
                input_time[0:49,:] = features_tensor[i+batch+j-20,node_set,:]
                j=17
                input_time[49:98,:] = features_tensor[i+batch+j-20,node_set,:]
                j=19
                node_set.remove(node)
                input_space = features_tensor[i+batch+j-20,node_set,:]
                input_src = features_tensor[i+batch-1,node:node+1,:]
                minus_1_presentation = Encoder(input_space, input_time, input_src, input_src, input_space, input_time)
                '''second half-------------------------------'''
                input_time = torch.zeros(49*2, 20)
                input_space = torch.zeros(48, 20)
                node_set = [x for x in range(49)]
                j = 25
                input_time[0:49, :] = features_tensor[i + batch + j - 20, node_set, :]
                j = 23
                input_time[49:98, :] = features_tensor[i + batch + j - 20, node_set, :]
                j = 21
                node_set.remove(node)
                input_space = features_tensor[i + batch + j - 20, node_set, :]
                input_src = features_tensor[i + batch + 1, node:node+1, :]
                plus_1_presentation = Encoder(input_space, input_time,
                                                       input_src, input_src, input_space, input_time)
                graph_present = torch.cat((minus_1_presentation,plus_1_presentation),1).squeeze()
                output = Decoder(graph_present)
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
            input_time_previous = torch.zeros(1, 20)
            input_time = torch.zeros(49 * 2, 20)
            input_space = torch.zeros(48, 20)
            '''first half-------------------------------'''
            input_time_current = torch.zeros(49, 20)
            node_set = [x for x in range(49)]
            j = 15
            input_time[0:49, :] = features_tensor[i + j - 20, node_set, :]
            j = 17
            input_time[49:98, :] = features_tensor[i + j - 20, node_set, :]
            j = 19
            node_set.remove(node)
            input_space = features_tensor[i + j - 20, node_set, :]
            input_src = features_tensor[i - 1, node:node+1, :]
            minus_1_presentation = Encoder(input_space, input_time,
                                           input_src, input_src, input_space, input_time)
            '''second half-------------------------------'''
            input_time_current = torch.zeros(49, 20)
            node_set = [x for x in range(49)]
            j = 25
            input_time[0:49, :] = features_tensor[i + j - 20, node_set, :]
            j = 23
            input_time[49:98, :] = features_tensor[i + j - 20, node_set, :]
            j = 21
            node_set.remove(node)
            input_space = features_tensor[i + j - 20, node_set, :]
            input_src = features_tensor[i + 1, node:node+1, :]
            plus_1_presentation = Encoder(input_space, input_time,
                                          input_src, input_src, input_space, input_time)
            graph_present = torch.cat((minus_1_presentation, plus_1_presentation), 1).squeeze()
            output = Decoder(graph_present)

            y = features_tensor[i, node, -1]
            loss += (output - y).pow(2).sum()
            N += 1

        valid_pred[i-1000, node] = output.item()
    Loss = (valid_pred - features_tensor[1000:1550, :, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP25_best_pred"+str(epoch))