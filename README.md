# Spatio-Temporal-SuperResolution-
GNNs with temporal data

介绍任务
介绍应用
介绍数据

1. Exp1
Fixed sliding window on temporal data, a GCN is applied on each time frame.


2. Exp2
A RNN is applied on temporal data, with the input from the output of a GCN, which is applied on each time frame.


3. Exp3
Random walk on temporal graph to get the representation. A DNN is applied with the representation as input and the node label as output.


4. Exp4
Random walk on temporal graph and embed node similar to the node embedding in GraphSAGE.

