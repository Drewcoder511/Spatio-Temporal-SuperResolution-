from Linear import LinearEncode
import torch
import numpy as np
from Train2 import train
from Train2 import random_walk_path
import copy
# old randomwalk_fixed
torch.set_default_tensor_type(torch.DoubleTensor)

walk_num = 20
walk_len = 10
Model = LinearEncode(walk_num*walk_len*19, 30, 5, 1)

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')
PATH = "/Users/huoshengming/Downloads/window_best"

for i in range(18):
    data = features[:,:,i].reshape(-1)
    features[:,:,i] = features[:,:,i] - np.mean(data)
    features[:,:,i] = features[:,:,i]/np.max(np.abs(data))

features_tensor = torch.from_numpy(features)

best_valid_loss = np.inf

walk_path_all = random_walk_path(adj_mat, features_tensor, 0.5)

'''
for epoch in range(50):
    best_model, trainloss, validloss = train(features_tensor, Model, walk_path_all)
    if validloss < best_valid_loss:
        best_valid_loss = validloss
        print("save!")
        torch.save(Model.state_dict(), PATH)
        window_best = copy.deepcopy(Model)'''
print(walk_path_all)

walk_path_all = random_walk_path(adj_mat, features_tensor)
print(walk_path_all)