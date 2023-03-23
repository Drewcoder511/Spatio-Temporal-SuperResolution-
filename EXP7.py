import numpy as np
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
Encoder = GCN(19, 6)
Decoder = GRU(12, hidden_size, 1, 1).to(device)

#*****************************************************************************************
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=5, walk_len=20)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=5, walk_len=20)
for i in walk_path_all1:
    walk_path_all.append(i)

time_node = {}
for i in walk_path_all:
    for j in i:
        if j[0] in time_node:
            if j[1] not in time_node[j[0]]:
                time_node[j[0]].append(j[1])
        else:
            time_node[j[0]] = [j[1]]
#*****************************************************************************************
best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

Loss_train = []
for epoch in range(10):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    node = 0
    for batch in train_set[node]:
        loss = 0
        for i in batch:  # each time frame
            # for node in range(0,49): # each node
            # 通过path找节点feature

            h0 = torch.zeros(1, 1, 4).to(device)  # 2 for bidirection
            for j in sorted(time_node):
                output = Encoder(features_tensor[j - 20 + i, 0, :], features_tensor[j - 20 + i, time_node[j], :])
                output = output.reshape(1, 1, -1).to(device)
                output, h0 = Decoder(output, h0)

            y = features_tensor[i, node, -1]
            loss += (output - y).pow(2).sum()
            N += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.item()
    Loss_train.append(Loss / N)
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))

    Encoder.eval()
    Decoder.eval()
    Loss = 0

    valid_pred = torch.zeros(550, 1)  # time, node
    for i in range(1000, 1550):
        node = 0
        h0 = torch.zeros(1, 1, 4).to(device)  # 2 for bidirection
        for j in sorted(time_node):
            output = Encoder(features_tensor[j - 20 + i, 0, :], features_tensor[j - 20 + i, time_node[j], :])
            output = output.reshape(1, 1, -1).to(device)
            output, h0 = Decoder(output, h0)

        valid_pred[i-1000, node] = torch.exp(output).item()
    Loss += (valid_pred - features_tensor[1000:1550, 0:1, -1]).pow(2).sum().item()

    print("epoch:", epoch + 1, "validation loss:", Loss/550)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP7_best_pred"+str(epoch))