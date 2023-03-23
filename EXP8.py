import numpy as np
import torch

from BiRNN import *
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()
for i in range(49):
    train_set[i] = np.array_split(train_set[i], 20)

features_tensor = torch.from_numpy(features)


for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

hidden_size = 4
batch_size = 20
Encoder = GCN(19, 7, aggr_neighbor_method="mean")
Decoder = Transformer(d_model=14, nhead=7, output_dim=1, batch_first=True)

#*****************************************************************************************
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=5, walk_len=20)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=5, walk_len=20)
for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

time_node = []
for node, walk_path_node in enumerate(walk_path_all):
    time_node.append({})
    for i in walk_path_node:
        for j in i:
            if j[0] in time_node[node]:
                if j[1] not in time_node[node][j[0]]:
                    time_node[node][j[0]].append(j[1])
            else:
                time_node[node][j[0]] = [j[1]]

time_node_index = []
for node, time_index in enumerate(time_node):
    time_node_index.append([6,6])
    for i in time_index.keys():
        if i < 20:
            time_node_index[node][0] -= 1
        else:
            time_node_index[node][1] -= 1
#*****************************************************************************************
best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.001, weight_decay=0.0005)

Loss_train = []
for epoch in range(10):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    for node in range(14,15):
        for batch in train_set[node]:
            loss = 0
            src = torch.zeros(len(batch), len(time_node[node]), 14)
            for bat, i in enumerate(batch):
                for index, j in enumerate(sorted(time_node[node])):
                    src[bat, index, :] = Encoder(features_tensor[j - 20 + i, 0, :],
                                                 features_tensor[j - 20 + i, time_node[node][j], :])
            mean_val = torch.mean(src)
            padding = (0,0,
                       time_node_index[node][0],time_node_index[node][1],
                       0,0)

            src = F.pad(src, padding, value=mean_val.item())
            prediction = Decoder(src, features_tensor[batch, 0, 0:18])
            y = features_tensor[batch, node:node+1, -1]
            loss = (prediction - y).pow(2).mean()
            N += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Loss += loss.item()
        print("epoch:", epoch + 1, "node", node+1, "train loss:{:.4f}".format(Loss / N))

    Encoder.eval()
    Decoder.eval()
    Loss = 0

    valid_pred = torch.zeros(550, 1)  # time, node
    node = 0

    for node in range(14,15):
        src = torch.zeros(550, len(time_node[node]), 14)
        for i in range(1000, 1550):
            for index, j in enumerate(sorted(time_node[node])):
                src[i-1000, index, :] = Encoder(features_tensor[j-20+i, 0, :],
                                                features_tensor[j-20+i, time_node[node][j], :])

        mean_val = torch.mean(src)
        padding = (0, 0,
                   time_node_index[node][0], time_node_index[node][1],
                   0, 0)
        src = F.pad(src, padding, value=mean_val.item())
        prediction = Decoder(src, features_tensor[1000:1550, node, 0:18])

        Loss += (prediction - features_tensor[1000:1550, node:node+1, -1]).pow(2).sum().item()

    print("epoch:", epoch + 1, "validation loss:", Loss/(550))
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(prediction, "/Users/huoshengming/Downloads/Best_model/EXP8_best_pred"+str(epoch))