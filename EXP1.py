from NodeEmbed import NodeEmbed
import torch
import numpy as np
from Linear import LinearEncode
from Train2 import random_walk_path
import matplotlib.pyplot as plt
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

walk_num = 20
walk_len = 10
Encoder = NodeEmbed(18, 10, 21, 8, 1)

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

features_tensor = torch.from_numpy(features)

for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()
for i in range(49):
    train_set[i] = np.array_split(train_set[i], 20)

model = LinearEncode(walk_num*walk_len*19, 30, 5, 1)

walk_path_all = random_walk_path(adj_mat, features_tensor)

# time, node, feature: (1577, 49, 19)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

best_valid_loss = float('inf')

for epoch in range(20):
    model.train()
    Loss = 0
    N = 0
    for node in range(0, 1):  # each node
        for batch in train_set[node]:
            loss = 0
            for i in batch:
                # 通过path找节点feature
                feature = torch.zeros(walk_len * walk_num, 19)

                for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                    for p in range(len(walk_path_all[node][path])):
                        time = walk_path_all[node][path][p][0] + i - 20
                        node = walk_path_all[node][path][p][1]
                        feature[path * walk_len + p] = features_tensor[time][node]

                feature = feature.reshape(-1)
                out = model(feature)
                y = features_tensor[i, node, -1]
                loss += (out - y).pow(2).sum()
                N += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    model.eval()

    prediction = torch.zeros(550)
    for i in range(1000, 1550):  # each time frame
        for node in range(0, 1):  # each node
            feature = torch.zeros(walk_len * walk_num, 19)
            for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                for p in range(len(walk_path_all[node][path])):
                    time = walk_path_all[node][path][p][0] + i - 20
                    node = walk_path_all[node][path][p][1]
                    feature[path * walk_len + p] = features_tensor[time][node]

            feature = feature.reshape(-1)
            prediction[i-1000] = model(feature)
    y = features_tensor[1000:1550, node, -1]
    Loss += ((prediction - y).pow(2).sum()).item()

    print("epoch:", epoch + 1, "validation loss:", Loss/550)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(prediction, "/Users/huoshengming/Downloads/Best_model/EXP1_best_pred"+str(epoch))
