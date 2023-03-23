import numpy as np
import copy
from Aggregator import *
torch.set_default_tensor_type(torch.DoubleTensor)

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

features_tensor = torch.from_numpy(features)


for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

walk_result = []
for node in range(49):
    walk_result.append(NeigAgg_3D(adj_mat, node, [4,4,4,4], features_tensor, 20,
                                  P=0.3, Graph_level=True, lower=1, upper=1, Feature_similarity=False))

Model = Decoder(19, 11, 8, 18, 1, 4)
# time, node, feature: (1577, 49, 19)
PATH = "/Users/huoshengming/Downloads/Best_model/best"
optimizer = torch.optim.Adam(Model.parameters(), lr=500)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
Loss_train = []
Loss_valid = []

best_valid_loss = float('inf')

for epoch in range(10):
    Model.train()
    Loss = 0
    N = 0
    for batch in range(49):  # 49 batches
        loss = 0
        for i in range(20 + batch * 20, 40 + batch * 20):  # each time frame
            for node in range(0, 49):  # each node
                # 通过path找节点feature
                result_feature = walk_to_feature(walk_result[node], i, features_tensor)
                out = Model(result_feature, features_tensor[i, node, 0:18])
                y = features_tensor[i, node, -1]
                #print("out",out)
                #print("y",y)
                loss += (out - y).pow(2).sum()
                N += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.item()
        # print("epoch:", epoch, batch,',Train Loss(Fixed walk)(49 node, 20 time stamp each bacth):{:.4f}'.format(loss))
    Loss_train.append(Loss / N)
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss / N))
    #scheduler.step()
    Model.eval()

    valid_pred = torch.zeros(550, 49)  # node, time
    for i in range(1000, 1550):
        for node in range(0, 49):
            # 通过path找节点feature
            result_feature = walk_to_feature(walk_result[node], i, features_tensor)
            valid_pred[i - 1000, node] = Model(result_feature, features_tensor[i, node, 0:18]).item()

    y = features_tensor[1000:1550, :, -1]
    Loss = ((valid_pred - y).pow(2).sum())

    Loss_valid.append(Loss)
    print("epoch:", epoch + 1, "validation loss:", Loss / N)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(Model.state_dict(), PATH)
        torch.save(valid_pred, "/Users/huoshengming/Downloads/Best_model/best_pred")


