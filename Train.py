import torch
import copy
from NodeEmbed import NodeEmbed
from RandomWalk02 import node2vecWalk_3D_Traffic2
import matplotlib.pyplot as plt

model = NodeEmbed(18, 10, 21, 8, 1)

def train(features_tensor, adj_mat, model, batches=49, lr = 0.01,
          walk_num = 20, walk_len = 10,
          PATH = "/Users/huoshengming/Downloads/old_fixedwalk_best"
          ):
    # old randomwalk_fixed

    # Decoder = NodeDecode(21, 8, 1)

    # time, node, feature: (1577, 49, 19)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    # 预加载walk_path

    walk_path_all = []

    for node in range(0, 49):  # each node

        walk_path = []  # length: walk_len*walk_num

        for walk in range(walk_num):  # each walk

            walk_path.append(
                node2vecWalk_3D_Traffic2(adj_mat, node, walk_len, features_tensor, 20,
                                         Graph_level=True, lower=1, upper=1, Feature_similarity=False))

        walk_path_all.append(walk_path)

    Loss_train = []
    Loss_valid = []
    best_valid_loss = float('inf')
    fig, ax = plt.subplots()
    epoch_list = []
    for epoch in range(epochs):

        model.train()

        Loss = 0
        N = 0
        for batch in range(batches):  # 49 batches
            loss = 0
            for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame

                for node in range(0, 49):  # each node
                    # 通过path找节点feature
                    src_node_feature = features_tensor[i][node][0:18]

                    neig = torch.zeros(walk_len * walk_num, 19)

                    for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                        for p in range(len(walk_path_all[node][path])):
                            time = walk_path_all[node][path][p][0] + i - 20
                            node = walk_path_all[node][path][p][1]
                            neig[path * walk_len + p] = features_tensor[time][node]

                    neig_node_feature = neig[:, 0:18]
                    neig_node_value = neig[:, 18]

                    # print(src_node_feature, neig_node_feature.mean(dim=0) , neig_node_value.mean())

                    out = model(src_node_feature, neig_node_feature, neig_node_value)

                    y = features_tensor[i, node, -1]

                    loss += (out - y).pow(2).sum()
                    N += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()
        Loss_train.append(Loss / N)
        print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

        model.eval()
        validloss = 0
        N = 0
        for i in range(1000, 1550):  # each time frame
            for node in range(0, 49):  # each node
                src_node_feature = features_tensor[i][node][0:18]

                neig = torch.zeros(walk_len * walk_num, 19)

                for path in range(len(walk_path_all[node])):  # walk_path:[t, node]
                    for p in range(len(walk_path_all[node][path])):
                        time = walk_path_all[node][path][p][0] + i - 20
                        node = walk_path_all[node][path][p][1]
                        neig[path * walk_len + p] = features_tensor[time][node]

                neig_node_feature = neig[:, 0:18]
                neig_node_value = neig[:, 18]

                out = model(src_node_feature, neig_node_feature, neig_node_value)

                y = features_tensor[i, node, -1]

                validloss += ((out - y).pow(2).sum()).item()
                N += 1

        Loss_valid.append(validloss / N)
        print("epoch:", epoch + 1, "validation loss:", validloss / N)
        epoch_list.append(epoch + 1)
        ax.cla()
        ax.plot(epoch_list, Loss_train)
        ax.plot(epoch_list, Loss_valid)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.pause(0.1)


    return fixedwalk_best
