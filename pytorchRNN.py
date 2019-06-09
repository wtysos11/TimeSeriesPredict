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
unit = 10
factor = 0.01
w = math.pi/unit
b1 = unit*w
b2 = 2*b1
x1 = w*np.arange(0,unit,factor)
x2 = w*np.arange(unit,2*unit,factor)+b1
x3 = w*np.arange(2*unit,3*unit,factor)+b2
y1 = np.array(list(map(math.sin,x1)))
y2 = 2*np.array(list(map(math.sin,x2)))
y3 = 4*np.array(list(map(math.sin,x3)))
y = np.concatenate((y1,y2,y3))
y_true = y
x = np.arange(0,3*unit,factor)

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

X_train = x[:2800]
X_test = x[2800:]
y_train = y[:2800]
y_test = y[2800:]

X_train = np.reshape(X_train,(len(X_train),1,w))
X_test = np.reshape(X_test,(len(X_test),1,w))

device ='cpu'

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

input_size = 48
hidden_size = 32
num_layers = 3
num_classes = 1
batch_size = 1
learning_rate = 0.01
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.MSELoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
    if i%10 == 0:
        print(images,label,outputs,loss)