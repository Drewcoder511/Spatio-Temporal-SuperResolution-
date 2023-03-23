import numpy as np
from LinearInterpolation import *
from BiRNN import *
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

adj_mat = np.load('/Users/huoshengming/Downloads/graph-mtx-processd.npy')
features = np.load('/Users/huoshengming/Downloads/node-values.npy')

train_set = np.load('/Users/huoshengming/Downloads/Best_model/train_set.npy', allow_pickle=True).item()
valid_set = np.load('/Users/huoshengming/Downloads/Best_model/valid_set.npy', allow_pickle=True).item()
for i in range(49):
    train_set[i] = np.array_split(train_set[i], 20)

features_tensor = torch.from_numpy(features)


for i in range(18):
    data = features_tensor[:,:,i].reshape(-1)
    features_tensor[:,:,i] = features_tensor[:,:,i] - torch.mean(data)
    features_tensor[:,:,i] = features_tensor[:,:,i]/torch.max(torch.abs(data))

hidden_size = 4
batch_size = 20
model = LinearInt(16, 4, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

for epoch in range(10):
    model.train()

    Loss = 0
    N = 0
    for i in range(20, 1000, 20):
        loss = 0
        for batch in range(20):
            for node in range(0,1):
                eight_range = torch.zeros(16)
                for j in range(0,16):
                    eight_range[j] = features_tensor[i+batch+2*j-15,node,-1]
                prediction = model(eight_range)
                y = features_tensor[i+batch,node,-1]
                loss += (prediction - y).pow(2).sum()
                N += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss += loss.item()
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss/N))

    model.eval()
    prediction = torch.zeros(530)
    for i in range(1000, 1530):
        for node in range(0, 1):
            eight_range = torch.zeros(16)
            for j in range(0, 16):
                eight_range[j] = features_tensor[i + 2 * j - 15, node, -1]
            prediction[i-1000] = model(eight_range).item()
    y = features_tensor[1000:1530, node, -1]
    Loss = (prediction - y).pow(2).sum().item()

    print("epoch:", epoch + 1, "validation loss:", Loss/530)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
    torch.save(prediction, "/Users/huoshengming/Downloads/Best_model/EXP9_best_pred"+str(epoch))