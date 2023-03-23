'''
Node Embedding: Dimension: 18
Concatenate Node Embedding + Node Label: 18 + 1 = 19
Encoder: SageGCN: 19 to 6 (Without center node feature because of full connected graph)
Decoder: GRU: 6 to 20 to 1 (With all zero input as h0 input for the first h0)

Validation loss: 0.00030
'''


import numpy as np
from BiRNN import *
from gensim.models import Word2Vec
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/windmill_large_adjmat.npy')
features = np.load('/Users/huoshengming/Downloads/windmill_large.npy')

features_tensor = torch.from_numpy(features)

walk_num = 50
walk_len = 20
walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=walk_num, walk_len=walk_len, src_node=True)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.6, walk_num=walk_num, walk_len=walk_len, src_node=True)

for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

time_node = {}
for i in walk_path_all[0]:
    for j in i:
        if j[0] in time_node:
            if j[1] not in time_node[j[0]]:
                time_node[j[0]].append(j[1])
        else:
            time_node[j[0]] = [j[1]]

hidden_size = 4
batch_size = 20
Encoder = GCN2(19, 6)
Decoder = GRU(6, hidden_size, 1, 1).to(device)

#****************************************************************************************
for node, i in enumerate(walk_path_all1):
    for j in i:
        walk_path_all[node].append(j)

time_node = {}
for i in walk_path_all[0]:
    for j in i:
        if j[0] in time_node:
            if j[1] not in time_node[j[0]]:
                time_node[j[0]].append(j[1])
        else:
            time_node[j[0]] = [j[1]]
#*****************************************************************************************
# node embedding model
walk = []
for i in walk_path_all:
    for j in i:
        sentence = [str(word) for word in j]
        walk.append(sentence)

model = Word2Vec(walk, vector_size=18, min_count=1)
wv = model.wv # word vector
del model

input_feature = {}
for j in sorted(time_node):
    input_feature[j] = torch.tensor(wv[time_node[j]])


best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(10):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    node = 0
    for i in range(100,10000,100):
        loss = 0
        for batch in range(100):  # each time frame

            h0 = torch.zeros(1, 1, 4).to(device)  # 2 for bidirection
            for j in sorted(time_node):
                Input = torch.cat((input_feature[j], torch.unsqueeze(features_tensor[i+batch+j-20,time_node[j]], 1)), dim=1)
                output = Encoder(Input)
                output = output.reshape(1, 1, -1).to(device)
                output, h0 = Decoder(output, h0)

            y = features_tensor[i+batch, node]
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

    valid_pred = torch.zeros(7000, 1)  # time, node
    for i in range(10000, 17000):
        node = 0
        h0 = torch.zeros(1, 1, 4).to(device)  # 2 for bidirection
        for j in sorted(time_node):
            Input = torch.cat((input_feature[j], torch.unsqueeze(features_tensor[i+j-20,time_node[j]], 1)), dim=1)
            output = Encoder(Input)
            output = output.reshape(1, 1, -1).to(device)
            output, h0 = Decoder(output, h0)

        valid_pred[i-10000, node] = output.item()
    Loss += (valid_pred - features_tensor[10000:17000, 0:1]).pow(2).sum().item()

    print("epoch:", epoch + 1, "validation loss:", Loss/7000)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP13_best_pred"+str(epoch))