'''
Dataset: Aus Weather
Node Dimension: 19 (No Embedding)

Encoder: SageGCN: 19 to 6 (Without center node feature because of full connected graph)
Decoder: Linear: 6 to 20 to 1 (With all zero input as h0 input for the first h0)

Validation loss: 27.6
'''

import numpy as np
from BiRNN import *
from gensim.models import Word2Vec

###

from BiRNN import *
from gensim.models import Word2Vec
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()

for i in range(18):
    data = features[:,:,i].reshape(-1)
    features[:,:,i] = features[:,:,i] - np.mean(data)
    features[:,:,i] = features[:,:,i]/np.max(np.abs(data))

features_tensor = torch.from_numpy(features)

'''
Add time label in the data:
now the data shape is [1577,49,20]
'''
time_label = torch.unsqueeze(torch.unsqueeze(torch.arange(len(features_tensor)),dim=1),dim=1)
time_label = time_label.expand(-1,features_tensor.shape[1],1)
features_tensor = torch.cat((time_label, features_tensor), 2)

hidden_size = 4
batch_size = 20
Encoder = GCN2(20, 6)
Decoder = Lin(6*6, 10, 1)
#*****************************************************************************************
best_valid_loss = float('inf')
optimizer = torch.optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.01, weight_decay=0.0005)

for epoch in range(50):
    Encoder.train()
    Decoder.train()

    Loss = 0
    N = 0
    for i in range(20,1000,20):
        for node in range(49):
            loss = 0
            for batch in range(20):  # each time frame
                src = torch.zeros(6, 6)
                for j in [15, 17, 19, 21, 23, 25]:
                    Input = features_tensor[i+batch+j-20,:,:]
                    output = Encoder(Input)
                    output = output.reshape(1, 1, -1)
                    src[int((j-15)/2), :] = output

                src = src.reshape(-1)
                output = Decoder(src)

                y = features_tensor[i+batch, node, -1]
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

    valid_pred = torch.zeros(550, 49)  # time, node
    for node in range(49):
        for i in range(1000, 1550):
            src = torch.zeros(6, 6)
            for j in [15, 17, 19, 21, 23, 25]:
                Input = features_tensor[i + j - 20, :, :]
                output = Encoder(Input)
                output = output.reshape(1, 1, -1)
                src[int((j - 15) / 2), :] = output

            src = src.reshape(-1)
            output = Decoder(src)

            src = src.reshape(-1)
            output = Decoder(src)

            valid_pred[i-1000, node] = output.item()
    Loss += (valid_pred - features_tensor[1000:1550, :, -1]).pow(2).mean().item()

    print("epoch:", epoch + 1, "validation loss:", Loss)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP27_best_pred"+str(epoch))