import numpy as np
import torch
from RandomWalk02 import node2vecWalk_3D_Traffic2

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

features_tensor = torch.from_numpy(features)

walk_num = 20
walk_len = 10
node = 10

print(node2vecWalk_3D_Traffic2(adj_mat, node, walk_len, features_tensor, 20,
                         Graph_level=True, lower=1, upper=1, Feature_similarity=False))