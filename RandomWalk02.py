import numpy as np


def Edge_to_Adjdict(edge_index, node_num):  # Transfer Edge index to dictionary
    # edge_index: tensor([[ 0, 0, 0, 1, 1], [ 1, 12, 16, 0, 20]])
    # adjdict:[[tensor(1), tensor(12), tensor(16)],[tensor(0), tensor(20)]]

    adjdict = []
    for i in range(node_num):  # 0, 1, 2, ..., node_num
        adjdict.append([])
        for j in range(len(edge_index[0])):
            if edge_index[0][j] == i:
                adjdict[i].append(edge_index[1][j])

    return adjdict


# adj_mat[node_0, node_1]

def node2vecWalk_3D_Traffic2(adj_mat, start_node, l, data, t, P=0.3, Graph_level=False, lower=1, upper=1, src_node=False,
                             Feature_similarity=False):  # include node features

    # edge_index:
    # start_node: the node to start random walk

    # l: length of random walk

    # features: features of time t(features[t][node_num])
    # data (data[t].x, data[t].y)

    # t: time frame of the starting node
    # 删掉了p，q，因为49节点问题没有距离为0和1的节点（全联接）
    # 随时间相隔越来越远的概率衰减函数是 0.5 + 1/t
    if src_node:
        walk = [[t, start_node]]
    else:
        walk = []

    gap = lower + upper  # The gap to skip

    nearby = []

    probability = []

    distance_1_time = [t + upper, t - lower]  # dist(start_node, i)=1

    if Graph_level:  # Graph level prediction so no random walk on frame t
        if t - lower < 0:  # if start_node at the first frame
            Dis_1_time = [t + upper]
        elif t + upper > len(data) - 1:  # if start_node at the last frame
            Dis_1_time = [t - lower]
        else:
            Dis_1_time = [t - lower, t + upper]

    # random choose one node as the first step
    pro = adj_mat[start_node]  # 49 nodes

    curnbr = np.arange(len(adj_mat))

    pro = pro / np.sum(pro)  # normalization probabilities

    if len(Dis_1_time) == 1:
        time = Dis_1_time[0]
    else:
        time = np.random.choice(np.array(Dis_1_time), p=np.array([0.5, 0.5]))  # 1/2 pb choose one time

    node = start_node

    walk.append([time, node])  ### Dis_1变1:n的数组然后再索引回来

    # start to walk
    while len(walk) < l:
        [current_t, current_node] = walk[-1]  # the current node with t
        if Graph_level:  # Graph level prediction so no random walk on frame t
            if current_t - gap < 0:  # cannot walk down through t
                Nbr_with_t = [current_t, current_t + gap]
                pro_t = np.zeros(2)
                # stay
                pro_t[0] = 1 / abs(Nbr_with_t[0] - t) + P
                # jump high
                pro_t[1] = 1 / abs(Nbr_with_t[1] - t)


                pro_t = pro_t / np.sum(pro_t)  # norm

                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)
                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node

            elif current_t + gap > len(data) - 1:  # cannot walk up through t
                Nbr_with_t = [current_t, current_t - gap]
                pro_t = np.zeros(2)
                # stay
                pro_t[1] = 1 / abs(Nbr_with_t[0] - t) + P  # more perfer to stay at current time
                # jump low
                pro_t[2] = 1 / abs(Nbr_with_t[1] - t)

                pro_t = pro_t / np.sum(pro_t)  # norm
                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)

                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node

            else:
                Nbr_with_t = [current_t - gap, current_t, current_t + gap]
                pro_t = np.zeros(3)
                # jump low
                pro_t[0] = 1 / abs(Nbr_with_t[0] - t)
                # stay
                pro_t[1] = 1 / abs(Nbr_with_t[1] - t) + P # more perfer to stay at current time
                # jump high
                pro_t[2] = 1 / abs(Nbr_with_t[2] - t)

                pro_t = pro_t / np.sum(pro_t)  # norm
                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)

                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node
        walk.append([time, node])
    return walk

def node2vecWalk_3D_Traffic3(adj_mat, start_node, l, data, t, P=0.3, Graph_level=False, lower=1, upper=1, src_node=False,
                             Feature_similarity=False):  # include node features

    # edge_index:
    # start_node: the node to start random walk

    # l: length of random walk

    # features: features of time t(features[t][node_num])
    # data (data[t].x, data[t].y)

    # t: time frame of the starting node
    # 删掉了p，q，因为49节点问题没有距离为0和1的节点（全联接）
    # 随时间相隔越来越远的概率衰减函数是 0.5 + 1/t
    if src_node:
        walk = [[t, start_node]]
    else:
        walk = []

    gap = lower + upper  # The gap to skip

    nearby = []

    probability = []

    distance_1_time = [t + upper, t - lower]  # dist(start_node, i)=1

    if Graph_level:  # Graph level prediction so no random walk on frame t
        if t - lower < 0:  # if start_node at the first frame
            Dis_1_time = [t + upper]
        elif t + upper > len(data) - 1:  # if start_node at the last frame
            Dis_1_time = [t - lower]
        else:
            Dis_1_time = [t - lower, t + upper]

    # random choose one node as the first step
    pro = adj_mat[start_node]  # 49 nodes

    curnbr = np.arange(len(adj_mat))

    pro = pro / np.sum(pro)  # normalization probabilities

    if len(Dis_1_time) == 1:
        time = Dis_1_time[0]
    else:
        time = np.random.choice(np.array(Dis_1_time), p=np.array([0.5, 0.5]))  # 1/2 pb choose one time

    node = start_node

    walk.append([time, node])  ### Dis_1变1:n的数组然后再索引回来

    # start to walk
    while len(walk) < l:
        [current_t, current_node] = walk[-1]  # the current node with t
        if Graph_level:  # Graph level prediction so no random walk on frame t
            if current_t - gap < 0:  # cannot walk down through t
                Nbr_with_t = [current_t, current_t + gap]
                pro_t = np.zeros(2)
                # stay
                pro_t[0] = abs(Nbr_with_t[0] - t) + P
                # jump high
                pro_t[1] = abs(Nbr_with_t[1] - t)


                pro_t = pro_t / np.sum(pro_t)  # norm

                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)
                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node

            elif current_t + gap > len(data) - 1:  # cannot walk up through t
                Nbr_with_t = [current_t, current_t - gap]
                pro_t = np.zeros(2)
                # stay
                pro_t[1] = abs(Nbr_with_t[0] - t) + P  # more perfer to stay at current time
                # jump low
                pro_t[2] = abs(Nbr_with_t[1] - t)

                pro_t = pro_t / np.sum(pro_t)  # norm
                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)

                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node

            else:
                Nbr_with_t = [current_t - gap, current_t, current_t + gap]
                pro_t = np.zeros(3)
                # jump low
                pro_t[0] = abs(Nbr_with_t[0] - t)
                # stay
                pro_t[1] = abs(Nbr_with_t[1] - t) + P # more perfer to stay at current time
                # jump high
                pro_t[2] = abs(Nbr_with_t[2] - t)

                pro_t = pro_t / np.sum(pro_t)  # norm
                time = np.random.choice(np.array(Nbr_with_t), p=pro_t)

                if time == current_t:
                    pro = adj_mat[current_node]
                    pro = pro / np.sum(pro)
                    node = np.random.choice(curnbr, p=pro)
                else:
                    node = current_node
        walk.append([time, node])
    return walk
