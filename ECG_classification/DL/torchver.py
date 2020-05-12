import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy

import time

start = time.time()  # 시작 시간 저장


x_train = np.array(x_train)
x_test = np.array(x_test)


x_train_st = []
for i in range(len(x_train) - 2):
    data = np.hstack((x_train[i], x_train[i + 1], x_train[i + 2]))
    x_train_st.append(data)

x_train_st = np.array(x_train_st)

x_test_st = []
for i in range(len(x_test) - 2):
    data = np.hstack((x_test[i], x_test[i + 1], x_test[i + 2]))
    x_test_st.append(data)

x_test_st = np.array(x_test_st)

x_train = x_train_st[:, :, np.newaxis]
x_test = x_test_st[:, :, np.newaxis]
#
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train_st = []
for i in range(len(y_train) - 2):
    data = np.hstack((y_train[i], y_train[i + 1], y_train[i + 2]))
    y_train_st.append(data)

y_train_st = np.array(y_train_st)

y_test_st = []
for i in range(len(y_test) - 2):
    data = np.hstack((y_test[i], y_test[i + 1], y_test[i + 2]))
    y_test_st.append(data)

y_test_st = np.array(y_test_st)

for i in range(len(y_test_st)):
    if sum(y_test_st[i]) == 0:
        y_test_st[i] = [0]
    else:
        y_test_st[i] = [1]

for i in range(len(y_train_st)):
    if sum(y_train_st[i]) == 0:
        y_train_st[i] = [0]
    else:
        y_train_st[i] = [1]

y_train = y_train_st[0:len(y_train_st), 0:1]
y_test = y_test_st[0:len(y_test_st), 0:1]


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

x_train = torch.from_numpy(x_train).float().permute(0, 2, 1)
x_test = torch.from_numpy(x_test).float().permute(0, 2, 1)

y_train = torch.from_numpy(y_train).long()
y_test = torch.from_numpy(y_test).long()

batch_sizes = 16

train = TensorDataset(x_train, y_train)
train_loader = DataLoader(train, batch_size=batch_sizes, shuffle=True)
test = TensorDataset(x_test, y_test)
test_loader = DataLoader(test, batch_size=batch_sizes, shuffle=True)


class DetectionNet(nn.Module):
    def __init__(self):
        super(DetectionNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2)
        )
        self.layer9 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding_mode='zeros', padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )
        self.layer10 = nn.Sequential(
            nn.LSTM(3, 64, 1)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(16384, 2),
            nn.LogSoftmax()
        )

    def forward(self, out):
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out, _ = self.layer10(out)
        out = self.flatten(out)
        out = self.fc1(out)
        return out


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # if torch.cuda.is_available() else 'cpu'

model = DetectionNet().to(device)
num_epochs = 1
from torchsummaryX import summary

summary(model, torch.zeros(16, 1, 510).to(device))

print(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train.squeeze())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % batch_sizes == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

from sklearn.metrics import accuracy_score

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for x_test, y_test in test_loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        total += y_test.size(0)
        correct += (predicted == y_test.squeeze()).sum().item()

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
