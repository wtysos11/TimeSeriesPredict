#使用pytorch实现RNN预测网络流量数据，发现基准线在多次训练后偏移，以及无法合适地预测突变数据，效果较差

import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os
os.chdir('E:\\code\\homework\\timeseries\\webtrafic_lstm')
data = pd.read_csv('sav_2013_2017.csv')
y = data['hits'].values
#进行标准化操作，将时序数据y转变为x_standard

x_mean = np.mean(y)
x_std = np.std(y)
x_standard = (y-x_mean)/x_std

#按照窗宽划分每个时间步的输入

w = 48
x = []
for t in range(len(x_standard)-w):
    a = x_standard[t:t+w]
    x.append(a)

# 输入x和输出y

x = np.array(x)
y = x_standard[w:len(x_standard)+1]

x_origin = x
x_origin = np.reshape(x_origin,(len(x_origin),1,w))
y_origin = y

split_factor = 0.8
split_row = int(len(x) * split_factor)
print(split_row)
X_train, X_test = x[:split_row], x[split_row:]
y_train, y_test = y[:split_row], y[split_row:]

X_train = np.reshape(X_train,(len(X_train),1,w))
X_test = np.reshape(X_test,(len(X_test),1,w))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)# input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

device ='cpu'
input_size = 48
hidden_size = 32
num_layers = 3
num_classes = 1
batch_size = 20
learning_rate = 0.01
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(batch_size):
    for i in range(len(X_train)):
        images = X_train[i,:]
        label = y_train[i]
        test = torch.from_numpy(images[np.newaxis,:]).float()
        outputs = model(test)
        ans = torch.tensor([label])
        outputs = outputs.reshape(1)
        loss = criterion(outputs,ans)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%1000 == 0:
            print('batch:{}/loop:{} ans: {}, output:{} , loss:{}'.format(t,i,ans,outputs,loss.item()))

y_true = []
with torch.no_grad():
    for i in range(len(X_test)):
        images = X_test[i,:]
        label = y_test[i]
        test = torch.from_numpy(images[np.newaxis,:]).float()
        outputs = model(test)
        ans = torch.tensor(label)
        outputs = outputs.reshape(1)
        loss = criterion(outputs,ans)
        y_true.append(outputs.item())
        if i%1000==0:
            print('{} ans: {}, output:{} , loss:{}'.format(i,ans,outputs,loss.item()))

plt.figure()
plt.plot(y_test, color='blue',label='Original')
plt.plot(y_true, color='red',label='Prediction')
plt.legend(loc='best')
plt.title('Test - Comparison')
plt.show()

y_true = []
with torch.no_grad():
    for i in range(len(X_train)):
        images = X_train[i,:]
        label = y_train[i]
        test = torch.from_numpy(images[np.newaxis,:]).float()
        outputs = model(test)
        ans = torch.tensor(label)
        outputs = outputs.reshape(1)
        loss = criterion(outputs,ans)
        y_true.append(outputs.item())
        if i%1000==0:
            print('{} ans: {}, output:{} , loss:{}'.format(i,ans,outputs,loss.item()))

plt.figure(figsize=(20,10))
plt.plot(y_train, color='blue',label='Original')
plt.plot(y_true, color='red',label='Prediction')
plt.legend(loc='best')
plt.title('Test - Comparison')
plt.show()