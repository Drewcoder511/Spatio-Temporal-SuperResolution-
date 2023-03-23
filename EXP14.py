'''
Slice Window:
Input dimension: 319 node * 1 node label * 6 timestamp (-5, -3, -1, +1, +3, +5 timestamp)
Output dimension: 1

Validation loss:
'''


from Linear import LinearEncode
import torch
import numpy as np
import copy
import numpy as np
from BiRNN import *
from gensim.models import Word2Vec
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

adj_mat = np.load('/Users/huoshengming/Downloads/windmill_large_adjmat.npy')
features = np.load('/Users/huoshengming/Downloads/windmill_large.npy')
features_tensor = torch.from_numpy(features)

model = LinearEncode(len(adj_mat)*6, 60, 8, 1)

features_tensor = torch.from_numpy(features)

best_valid_loss = np.inf

time_list = torch.tensor([-5,-3,-1,1,3,5])
adj_mat = torch.tensor(adj_mat)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

for epoch in range(10):
    model.train()

    Loss = 0
    N = 0
    for i in range(100,10000,100):  # 49 batches
        loss = 0
        for batch in range(100):  # each time frame
            for node in range(0, 1):  # each node
                # 通过path找节点feature
                feature = torch.zeros(6,len(adj_mat))
                for time in time_list:
                    feature[int((time+5)/2),:]=features_tensor[time,:]

                feature = feature.reshape(-1)
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

    Loss = 0
    N = 0
    valid_pred = torch.zeros(7000, 1)
    for i in range(10000, 17000):  # each time frame
        for node in range(0, 1):  # each node
            # 通过path找节点feature
            feature = torch.zeros(6, len(adj_mat))

            for time in time_list:
                feature[int((time + 5) / 2), :] = features_tensor[time, :]
            feature = feature.reshape(-1)

            valid_pred[i-10000, node] = model(feature).item()
    Loss = (valid_pred - features_tensor[10000:17000, 0:1]).pow(2).mean()


    print("Validation loss:", Loss)

    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/EXP14_best_pred" + str(epoch))