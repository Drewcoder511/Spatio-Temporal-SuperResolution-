'''
Dataset: Windmill Large

Encoder: SageGCN: 1 to 6 (Without center node feature because of full connected graph)
Decoder: Linear

Validation loss: (1 node): 0.0653
'''

import numpy as np
from BiRNN import *

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/windmill_large_adjmat.npy')
features = np.load('/Users/huoshengming/Downloads/windmill_large.npy')

features_tensor = torch.from_numpy(features)

walk_num = 50
walk_len = 20

walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=walk_num, walk_len=walk_len)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=walk_num, walk_len=walk_len)

for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

time_node = []
for node in range(319):
    time_node.append({})
    for i in walk_path_all[node]:
        for j in i:
            if j[0] > 12 and j[0] < 28:
                if j[0] in time_node[node]:
                    if j[1] not in time_node[node][j[0]]:
                        time_node[node][j[0]].append(j[1])
                else:
                    time_node[node][j[0]] = [j[1]]

hidden_size = 4
batch_size = 20
Encoder = GCN2(1, 6)
Decoder = Lin(8*6, 10, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(30):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0

    for i in range(100,10000,100):
        for node in range(319):
            loss = 0
            for batch in range(100):  # each time frame

                src = torch.zeros(8, 6)
                for index, j in enumerate(sorted(time_node[node])):
                    Input = torch.unsqueeze(features_tensor[i+batch+j-20,time_node[node][j]], 1)
                    output = Encoder(Input)
                    src[index, :] = output

                src = src.reshape(-1)
                output = Decoder(src)

                y = features_tensor[i+batch, node]
                loss += (output - y).pow(2).sum()
                N += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            Loss += loss.item()

    print("epoch:", epoch + 1, "train loss:{:.6f}".format(Loss / N))

    Encoder.eval()
    Decoder.eval()
    Loss = 0

    valid_pred = torch.zeros(7000, 319)  # time, node
    for node in range(319):
        for i in range(10000, 17000):
            src = torch.zeros(8, 6)
            for index, j in enumerate(sorted(time_node[node])):
                Input = torch.unsqueeze(features_tensor[i+j-20,time_node[node][j]], 1)
                output = Encoder(Input)
                src[index, :] = output
            src = src.reshape(-1)
            output = Decoder(src)
            valid_pred[i-10000, node] = output.item()

    Loss = (valid_pred - features_tensor[10000:17000, 0:319]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP19_best_pred"+str(epoch))