'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)

Encoder: SageGCN: 19 to 6 (Without center node feature because of full connected graph)
Decoder: Linear: 6 to 20 to 1 (With all zero input as h0 input for the first h0)

Validation loss: 27.6
'''

import numpy as np
from BiRNN import *
from gensim.models import Word2Vec
torch.set_default_tensor_type(torch.DoubleTensor)

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
#*****************************************************************************************
best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(50):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    for i in range(20,1000,20):
        for node in range(49):
            loss = 0
            for batch in range(20):  # each time frame
                src = torch.zeros(6, 6)
                for index, j in enumerate(sorted(time_node[node])):
                    Input = features_tensor[i+batch+j-20,time_node[node][j],:]
                    output = Encoder(Input)
                    output = output.reshape(1, 1, -1)
                    src[index, :] = output
                src = src.reshape(-1)

                output = Decoder(src)

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
    for node in range(49):
        for i in range(1000, 1550):
            src = torch.zeros(6, 6)
            for index, j in enumerate(sorted(time_node[node])):
                Input = features_tensor[i + j - 20, time_node[node][j], :]
                output = Encoder(Input)
                output = output.reshape(1, 1, -1)
                src[index, :] = output

            src = src.reshape(-1)
            output = Decoder(src)

            valid_pred[i-1000, node] = output.item()
    Loss += (valid_pred - features_tensor[1000:1550, :, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP18_best_pred"+str(epoch))