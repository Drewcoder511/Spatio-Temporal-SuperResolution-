from Linear import LinearEncode
import torch
import numpy as np
import copy
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

walk_num = 20
walk_len = 10
model = LinearEncode(3*2*49*21, 60, 8, 1)

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')
PATH = "/Users/huoshengming/Downloads/window_best"

for i in range(18):
    data = features[:,:,i].reshape(-1)
    features[:,:,i] = features[:,:,i] - np.mean(data)
    features[:,:,i] = features[:,:,i]/np.max(np.abs(data))

features_tensor = torch.from_numpy(features)

best_valid_loss = np.inf

batches=49

time_list = torch.tensor([-5,-3,-1,1,3,5])
adj_mat = torch.tensor(adj_mat)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

for epoch in range(20):
    model.train()
    #################################

    Loss = 0
    N = 0
    for batch in range(49):  # 49 batches
        loss = 0
        for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame
            for node in range(0, 1):  # each node
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
        # print("epoch:", epoch, batch,',Train Loss(Fixed walk)(49 node, 20 time stamp each bacth):{:.4f}'.format(loss))
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    model.eval()

    Loss = 0
    N = 0
    for i in range(1000, 1550):  # each time frame
        for node in range(0, 49):  # each node
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

            Loss += ((out - y).pow(2).sum()).item()
            N += 1

    ###########################
    for batch in range(batches):  # 49 batches
        loss = 0
        for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame
            for node in range(0, 49):  # each node
                # time, node, feature: (1577, 49, 19)
                input = torch.zeros(6, 49, 21)
                for j, t_index in enumerate(time_list):
                    input[j, :, 0:19] = features_tensor[i+t_index]

                for j in range(49):
                    input[:, j, 19] = time_list
                    input[:, j, 20] = adj_mat[node, j]
                input = input.reshape(-1)
                out = model(input)
                y = features_tensor[i, node, -1]
                loss += (out - y).pow(2).sum()
                N += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
    print("Train loss:{:.4f}".format(trainloss / N))

    model.eval()
    validloss = 0
    N = 0
    for i in range(1000, 1550):  # each time frame
        for node in range(0, 49):  # each node
            input = torch.zeros(6, 49, 21)
            for j, t_index in enumerate(time_list):
                input[j, :, 0:19] = features_tensor[i + t_index]

            for j in range(49):
                input[:, j, 19] = time_list
                input[:, j, 20] = adj_mat[node, j]

            input = input.reshape(-1)
            out = model(input)
            y = features_tensor[i, node, -1]
            validloss += ((out - y).pow(2).sum()).item()
            N += 1
    print("Validation loss:", validloss / N)

    if validloss < best_valid_loss:
        best_valid_loss = validloss
        print("save!")
        torch.save(model.state_dict(), PATH)
        window_best = copy.deepcopy(model)