from Linear import LinearEncode
import torch
import numpy as np
from Train2 import train
from Train2 import random_walk_path
from Train3 import train
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

walk_num = 5
walk_len = 10
#Model = GCN(19, 6, 1)

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


walk_path_all = random_walk_path(adj_mat, features_tensor, P=0.3, walk_num=5)
walk_path_all1 = random_walk_path(adj_mat, features_tensor, P=0.5, walk_num=5)
for i in walk_path_all1:
    walk_path_all.append(i)
#optimizer = torch.optim.Adam(Model.parameters(), lr=0.01, weight_decay=0.0001)


#best_model, trainloss, validloss = train(features_tensor, Model, walk_path_all, optimizer)
print(walk_path_all)

time_list = torch.tensor([-5,-3,-1,1,3,5])
adj_mat = torch.tensor(adj_mat)