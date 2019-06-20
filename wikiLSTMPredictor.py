import os
os.chdir('E:\\code\\homework\\timeseries\\web-traffic-time-series-forecasting')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train_1.csv').fillna(0)
train.head()

sum_set = pd.DataFrame(train[['Page']])
sum_set['total'] = train.sum(axis=1)
sum_set = sum_set.sort_values('total',ascending=False)
top_pages = sum_set.index[0:10]

#找到访问量最大的十个网站，并保存在data_list中
data_list = []
for _,index in enumerate(top_pages):
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[index,cols]
    data_list.append(data)
    days = [x for x in range(len(cols))]
    plt.plot(days,data)
    plt.show()

#信号产生器
import numpy as np
import math
import random
def getSignal(length):
    x = np.linspace(1,length,length)
    para = np.random.rand(3,10)
    A = para[0]
    w = para[1]
    b = para[2]
    y = np.zeros(len(x))
    for i,_ in enumerate(y):
        ans = 0
        for j in range(len(A)):
            ans += A[j] * math.sin(w[j]*x[i]+b[j])
        y[i] = ans
    return y

#普通的Keras LSTM
#Keras LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

hidden_size = 8
output_size = 1
batch_size = 8
window_size = 5
epoch_time = 100

data_list[0] = getSignal(500)

def getWindow(data,window_size):
    x = []
    for t in range(len(data)-window_size):
        a = data[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),1,window_size))
    return x

for i,index in enumerate(data_list):
    if i>=1:
        break
    begin = time.time()
    data = np.array(data_list[i],'f')
    #进行切分
    X_train = data[:split_num]
    y_train = data[1:split_num+1]
    X_test = data[split_num:-1]
    y_test = data[split_num+1:]
    xsc = MinMaxScaler()
    ysc = MinMaxScaler()
    X_train = np.reshape(X_train,(-1,1))
    y_train = np.reshape(y_train,(-1,1))
    X_train = xsc.fit_transform(X_train)
    y_train = ysc.fit_transform(y_train)
    X_train = np.reshape(X_train,(-1,1,1))
    
    x = getWindow(X_train,window_size)
    y_train = y_train[window_size:]
    #Keras LSTM
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(hidden_size, return_sequences=True, input_shape=(x.shape[1], window_size)))
    regressor.add(LSTM(hidden_size, return_sequences=True))
    regressor.add(LSTM(hidden_size))
    regressor.add(Dense(output_size))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False)
    #进入测试环节
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    inputs = np.reshape(inputs,(-1,1,1))
    x = getWindow(inputs,window_size)
    y_pred = regressor.predict(x)
    y_pred = ysc.inverse_transform(y_pred)
    end = time.time()
    #计时结束
    y_test = y_test[window_size:]
    print('total time for {} is {} s'.format(i,end-begin))
    plt.figure()
    plt.plot(y_test, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    print(mean_squared_error(y_test,y_pred))

#stateful LSTM
#Keras stateful LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

hidden_size = 8
output_size = 1
batch_size = 1
window_size = 5
ephch_time = 100

def getWindow(data,window_size):
    x = []
    for t in range(len(data)-window_size):
        a = data[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),1,window_size))
    return x

for i,index in enumerate(data_list):
    if i>=1:
        break
    begin = time.time()
    data = np.array(data_list[i],'f')
    #进行切分
    X_train = data[:split_num]
    y_train = data[1:split_num+1]
    X_test = data[split_num:-1]
    y_test = data[split_num+1:]
    xsc = MinMaxScaler()
    ysc = MinMaxScaler()
    X_train = np.reshape(X_train,(-1,1))
    y_train = np.reshape(y_train,(-1,1))
    X_train = xsc.fit_transform(X_train)
    y_train = ysc.fit_transform(y_train)
    X_train = np.reshape(X_train,(-1,1,1))
    x = getWindow(X_train,window_size)
    y_train = y_train[window_size:]
    #Keras LSTM
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(hidden_size, return_sequences=True, input_shape=(x.shape[1], window_size),batch_size = batch_size,stateful=True))
    regressor.add(LSTM(hidden_size, return_sequences=True,batch_size = batch_size,stateful=True))
    regressor.add(LSTM(hidden_size,batch_size = batch_size,stateful=True))
    regressor.add(Dense(output_size))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False)
    #进入测试环节
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    inputs = np.reshape(inputs,(-1,1,1))
    x = getWindow(inputs,window_size)
    y_pred = regressor.predict(x,batch_size=1)
    y_pred = ysc.inverse_transform(y_pred)
    end = time.time()
    #计时结束
    y_test = y_test[window_size:]
    print('total time for {} is {} s'.format(i,end-begin))
    plt.figure()
    plt.plot(y_test, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    print(mean_squared_error(y_test,y_pred))

#Keras online training without stateful
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

hidden_size = 8
output_size = 1
batch_size = 1
window_size = 20
epoch_time = 20
train_split = 0.4
train_num = int(len(data_list[0])*train_split)

#predict_back为在线训练时训练的间隔
predict_epoch_time = 10
predict_window = window_size

def getWindow(data,window_size):
    x = []
    for t in range(len(data)-window_size):
        a = data[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),1,window_size))
    return x

for i,index in enumerate(data_list):
    if i>=1:
        break
    begin = time.time()
    data = np.array(data_list[i],'f')
    scaler = MinMaxScaler()
    data = np.reshape(data,(-1,1))
    data = scaler.fit_transform(data)
    data = np.reshape(data,(-1))
    #进行切分
    X_train = data[:train_num]
    y_train = data[1:train_num+1]
    X_train = np.reshape(X_train,(-1,1,1))
    x = getWindow(X_train,window_size)
    y_train = y_train[window_size:]
    #Keras LSTM
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(hidden_size, return_sequences=True, input_shape=(x.shape[1], window_size)))
    regressor.add(LSTM(hidden_size, return_sequences=True))
    regressor.add(LSTM(hidden_size))
    regressor.add(Dense(output_size))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False)
    #进入在线训练以及测试环节
    predict_counter = 0
    y_pred = []
    for t in range(train_num+1,len(data)-1):
        predict_counter += 1
        #预测当前值
        #打包之前的内容去预测
        passX = data[t-window_size:t]
        passX = np.reshape(passX,(1,1,window_size))
        prediction = regressor.predict(passX)
        y_pred.append(prediction.item())
        #开始训练
        if predict_counter >= predict_window:
            training_data = getWindow(data[t-2*window_size:t],window_size)
            label = data[t-window_size+1:t+1]
            predict_counter = 0
            regressor.fit(training_data,label, batch_size = batch_size,epochs = predict_epoch_time)
            
    y_pred = np.array(y_pred)
    y_pred = np.reshape(y_pred,(-1,1))
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = np.reshape(y_pred,(-1))
    end = time.time()
    #计时结束
    data = np.reshape(data,(-1,1))
    data = scaler.inverse_transform(data)
    data = np.reshape(data,(-1))
    y_true = data[len(data)-len(y_pred):]
    print('total time for {} is {} s'.format(i,end-begin))
    plt.figure()
    plt.plot(y_true, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    print(mean_squared_error(y_true,y_pred))