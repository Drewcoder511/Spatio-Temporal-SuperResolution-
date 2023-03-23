import numpy as np
from LinearInterpolation import *
from BiRNN import *

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

stacked_target = np.load("/Users/huoshengming/Downloads/windmill_large.npy", allow_pickle=True)

model = LinearInt(16, 4, 1)

best_valid_loss = float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.0005)

for epoch in range(10):
    model.train()

    Loss = 0
    N = 0
    for i in range(20, 12000, 20):
        loss = 0
        for batch in range(20):
            for node in range(0,1):
                eight_range = torch.zeros(16)
                for j in range(0,16):
                    eight_range[j] = stacked_target[i+batch+2*j-15,node]
                prediction = model(eight_range)
                y = stacked_target[i+batch,node]
                loss += (prediction - y).pow(2).sum()
                N += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss += loss.item()
    print("epoch:", epoch + 1, "train loss:{:.4f}".format(Loss/N))

    model.eval()
    prediction = torch.zeros(5450)
    for i in range(12000, 17450):
        for node in range(0, 1):
            eight_range = torch.zeros(16)
            for j in range(0, 16):
                eight_range[j] = stacked_target[i + 2 * j - 15, node]
            prediction[i-12000] = model(eight_range).item()
    y = stacked_target[12000:17450, node]
    Loss = (prediction - y).pow(2).sum().item()

    print("epoch:", epoch + 1, "validation loss:", Loss/5450)
    if Loss < best_valid_loss:
        best_valid_loss = Loss
        print("save!")
        torch.save(prediction, "/Users/huoshengming/Downloads/Best_model/EXP10_best_pred"+str(epoch))