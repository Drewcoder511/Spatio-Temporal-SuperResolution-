import torch
from RandomWalk02 import node2vecWalk_3D_Traffic2
import matplotlib.pyplot as plt

def random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=20, walk_len=10):
    walk_path_all = []
    for node in range(len(adj_mat)):  # each node
        walk_path = []  # length: walk_len*walk_num
        for walk in range(walk_num):  # each walk
            walk_path.append(
                node2vecWalk_3D_Traffic2(adj_mat, node, walk_len, features_tensor, 20, P=P,
                                         Graph_level=True, lower=1, upper=1, Feature_similarity=False))
        walk_path_all.append(walk_path)
    return walk_path_all

def train(features_tensor, model, walk_path_all, batches=49, lr = 0.01, walk_num=20, walk_len=10):
    # time, node, feature: (1577, 49, 19)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    model.train()
    trainloss = 0
    N = 0
    for batch in range(batches):  # 49 batches
        loss = 0
        for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame
            for node in range(0, 49):  # each node
                neig = torch.zeros(walk_len * walk_num, 19)
                for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                    for p in range(len(walk_path_all[node][path])):
                        time = walk_path_all[node][path][p][0] + i - 20
                        node = walk_path_all[node][path][p][1]
                        neig[path * walk_len + p] = features_tensor[time][node]
                neig = neig.reshape(-1)
                out = model(neig)
                y = features_tensor[i, node, -1]
                loss += (out - y).pow(2).sum()
                N += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
    print("Train loss:{:.4f}".format(trainloss/N))

    model.eval()
    validloss = 0
    N = 0
    for i in range(1000, 1550):  # each time frame
        for node in range(0, 49):  # each node
            neig = torch.zeros(walk_len * walk_num, 19)
            for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                for p in range(len(walk_path_all[node][path])):
                    time = walk_path_all[node][path][p][0] + i - 20
                    node = walk_path_all[node][path][p][1]
                    neig[path * walk_len + p] = features_tensor[time][node]
            neig = neig.reshape(-1)
            out = model(neig)

            y = features_tensor[i, node, -1]
            validloss += ((out - y).pow(2).sum()).item()
            N += 1
    print("Validation loss:", validloss/N)

    return model, trainloss/N, validloss/N
