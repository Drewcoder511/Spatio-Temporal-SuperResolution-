from NodeEmbed import NodeEmbed
import torch
import numpy as np
from Linear import LinearEncode
from Train2 import random_walk_path
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

walk_num = 50
walk_len = 20

adj_mat = np.load('/Users/huoshengming/Downloads/windmill_large_adjmat.npy')
features = np.load('/Users/huoshengming/Downloads/windmill_large.npy')

features_tensor = torch.from_numpy(features)

model = LinearEncode(walk_num*walk_len*2, 30, 5, 1)

walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=walk_num, walk_len=walk_len)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=walk_num, walk_len=walk_len)
for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

'''
time_node = {}
for i in walk_path_all[0]:
    for j in i:
        if j[0] in time_node:
            if j[1] not in time_node[j[0]]:
                time_node[j[0]].append(j[1])
        else:
            time_node[j[0]] = [j[1]]

# time, node: (17472, 319)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

best_valid_loss = float('inf')

for epoch in range(10):
    model.train()
    Loss = 0
    N = 0
    for node in range(0, 1):  # each node
        for i in range(20,12000,20):
            loss = 0
            for batch in range(20):
                # 通过path找节点feature
                feature = torch.zeros(walk_len*walk_num*2)
                for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                    for p in range(len(walk_path_all[node][path])):
                        time = walk_path_all[node][path][p][0] +i +batch -20
                        node = walk_path_all[node][path][p][1]
                        feature[path*walk_len + p] = features_tensor[time][node]

                out = model(feature)
                y = features_tensor[i, node]
                loss += (out - y).pow(2).sum()
                N += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    model.eval()

    prediction = torch.zeros(5000)
    for i in range(12000, 17000):  # each time frame
        for node in range(0, 1):  # each node
            feature = torch.zeros(walk_len * walk_num*2)
            for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                for p in range(len(walk_path_all[node][path])):
                    time = walk_path_all[node][path][p][0] + i - 20
                    node = walk_path_all[node][path][p][1]
                    feature[path * walk_len + p] = features_tensor[time][node]

            prediction[i-12000] = model(feature)
    y = features_tensor[12000:17000, node]
    print((prediction - y).shape)
    Loss += ((prediction - y).pow(2).sum()).item()

    print("epoch:", epoch + 1, "validation loss:", Loss/5000)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(prediction, "/Users/huoshengming/Downloads/Best_model/EXP11_best_pred"+str(epoch))