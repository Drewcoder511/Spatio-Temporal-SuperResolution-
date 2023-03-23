import numpy as np
from BiRNN import *
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

features_tensor = torch.from_numpy(features)


for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

hidden_size = 4
batch_size = 20
Encoder = GCN(19, 6)
Decoder = BiRNN(12, hidden_size, 1, 1).to(device)
linear = Lin(32, 4, 1)

#*****************************************************************************************
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=5, walk_len=20)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=5, walk_len=20)
for i in walk_path_all1:
    walk_path_all.append(i)

time_node = [{}, {}]
for i in walk_path_all:
    for j in i:
        if j[0] < 20:
            if j[0] in time_node[0]:
                if j[1] not in time_node[0][j[0]]:
                    time_node[0][j[0]].append(j[1])
            else:
                time_node[0][j[0]] = [j[1]]
        if j[0] > 20:
            if j[0] in time_node[1]:
                if j[1] not in time_node[1][j[0]]:
                    time_node[1][j[0]].append(j[1])
            else:
                time_node[1][j[0]] = [j[1]]
#*****************************************************************************************

optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()) + list(linear.parameters()), lr=10)

Loss_train = []
for epoch in range(10):
    Encoder.train()
    Decoder.train()
    linear.train()

    Loss = 0
    N = 0
    for batch in range(49):  # 49 batches
        loss = 0
        for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame

            # for node in range(0,49): # each node
            # 通过path找节点feature
            node = 0
            h0 = torch.zeros(2, 1, 4).to(device)  # 2 for bidirection
            c0 = torch.zeros(2, 1, 4).to(device)
            for j in sorted(time_node[0]):
                output = Encoder(features_tensor[j - 20 + i, 0, :], features_tensor[j - 20 + i, time_node[0][j], :])
                output = output.reshape(1, 1, -1).to(device)
                output, (h0, c0) = Decoder(output, h0, c0)

            Input = torch.cat((h0.reshape(-1), c0.reshape(-1)), dim=0)

            h0 = torch.zeros(2, 1, 4).to(device)  # 2 for bidirection
            c0 = torch.zeros(2, 1, 4).to(device)
            for j in sorted(time_node[1], reverse=True):
                output = Encoder(features_tensor[j - 20 + i, 0, :], features_tensor[j - 20 + i, time_node[1][j], :])
                output = output.reshape(1, 1, -1).to(device)
                output, (h0, c0) = Decoder(output, h0, c0)

            Input = torch.cat((Input, h0.reshape(-1)), dim=0)
            Input = torch.cat((Input, c0.reshape(-1)), dim=0)
            print(Input)
            pred = linear(Input)
            y = features_tensor[i, node, -1]
            loss += (pred - y).pow(2).sum()
            N += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    Loss_train.append(Loss / N)
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))