from cnn import Net
from data_processing import process_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

train, test, val = process_data()
device = torch.device('cuda:0')
net = Net().to(device)
X = torch.Tensor([i[0] for i in train]).view(-1, 98, 98).to(device)
# Normalize data
X = X / 255.0
y = torch.Tensor([i[1] for i in train]).to(device)

def train_model(net):
    epochs = 1
    batch_size = 32

    # Specifying optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    for epoch in range(epochs):
        for i in tqdm(range(0, len(X), batch_size)):
            batch_X = X[i: i+batch_size].view(-1, 1, 98, 98).to(device)
            batch_y = y[i: i+batch_size].to(device)

            net.zero_grad()

            output = net(batch_X)
            loss = loss_function(output, batch_y)
            loss.backward()
            optimizer.step()

        print(f'Epoch: {epoch}\tLoss: {loss}')

def test_model(net):
    correct = 0
    total = 0
    X = torch.Tensor([i[0] for i in test]).view(-1, 98, 98).to(device)
    y = torch.Tensor([i[1] for i in test]).to(device)
    with torch.no_grad():
        for i in tqdm(range(len(X))):
            real_target = torch.argmax(y[i])
            output = net(X.view(-1, 1, 98, 98))[0]
            prediction = torch.argmax(output)

            if prediction == real_target:
                correct += 1
            total += 1
    print('Accuracy:', round(correct/total, 3) * 100)
