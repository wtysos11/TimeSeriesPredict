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

import keras
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

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
window_size = 20
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
    history = LossHistory()
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(hidden_size, return_sequences=True, input_shape=(x.shape[1], window_size)))
    regressor.add(LSTM(hidden_size, return_sequences=True))
    regressor.add(LSTM(hidden_size))
    regressor.add(Dense(output_size))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = batch_size, epochs = epoch_time, verbose = 0.2,shuffle=False,callbacks=[history])
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
    history.loss_plot('epoch')
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

#Keras online training without stateful。online training效果不是很好，可能是滑动窗口的问题
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

# 随机森林，裸在线训练（仅根据前一时刻预测下一时刻）
from sklearn.ensemble import RandomForestRegressor
window_size = 40
# RandomForestRegressor
for i,index in enumerate(data_list):
    begin = time.time()
    data = np.array(data_list[i],'f')
    sc = MinMaxScaler()
    data = np.reshape(data,(-1,1))
    data = sc.fit_transform(data)
    y_pred = []
    for t in range(len(data)-window_size-1):
        #进行训练
        regressor = RandomForestRegressor(n_estimators=window_size//2,n_jobs = -1)
        regressor.fit(data[t:t+window_size],data[t+1:t+window_size+1].ravel())
        #开始预测
        inputs = [data[t+window_size+1]]
        inputs = np.reshape(inputs,(-1,1))
        pred = regressor.predict(inputs)
        pred = pred.reshape(-1,1)
        pred = sc.inverse_transform(pred)
        y_pred.append(pred[0])
    
    data = sc.inverse_transform(data)
    y_test = data[window_size+1:]
    end = time.time()
    print('total time for {} is {}s'.format(i,end-begin))
    plt.plot(y_pred,label='predict')
    plt.plot(y_test,label='true')
    plt.legend(loc='upper right')
    plt.show()
    
#SVR，使用sliding window的非在线版本
#使用SVR进行普通的预测
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
def getBack(series,sc):
    series = np.reshape(series,(-1,1))
    series = sc.inverse_transform(series)
    series = np.reshape(series,(-1))
    return series

train_split = 0.8
train_num = int(len(data_list[0])*train_split)
for i,index in enumerate(data_list):
    #数据提取
    data = np.array(data_list[i],'f')
    sc = MinMaxScaler()
    data = np.reshape(data,(-1,1))
    data = sc.fit_transform(data)
    data = np.reshape(data,(-1))
    #切分
    X_train = data[:train_num]
    X_test = data[train_num:-1]
    y_train = data[1:train_num+1]
    y_test = data[train_num+1:]
    
    x = getWindow(X_train,window_size)
    x  = np.reshape(x,(x.shape[0],x.shape[-1]))
    y_train = y_train[window_size:]
    
    begin = time.time()
    clf = GridSearchCV(SVR(kernel='rbf',gamma='auto',C=1e2),cv=5,param_grid={"gamma":np.logspace(-2,2,5)},scoring='neg_mean_squared_error')
    clf.fit(x,y_train)
    print(clf.best_params_,clf.best_score_)
    
    x = getWindow(X_test,window_size)
    x  = np.reshape(x,(x.shape[0],x.shape[-1]))
    y_test = y_test[window_size:]
    y_pred = clf.predict(x)
    end = time.time()
    print('time :{}s'.format(end-begin))
    
    y_test = getBack(y_test,sc)
    y_pred = getBack(y_pred,sc)
    
    plt.figure()
    plt.plot(y_test, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    print(mean_squared_error(y_test,y_pred))
    #data back?
    data = np.reshape(data,(-1,1))
    data = sc.inverse_transform(data)
    data = np.reshape(data,(-1))
    
#在线版本的SVR
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
def getBack(series,sc):
    series = np.reshape(series,(-1,1))
    series = sc.inverse_transform(series)
    series = np.reshape(series,(-1))
    return series

train_split = 0.4
window_size = 100
train_num = int(len(data_list[0])*train_split)
for i,index in enumerate(data_list):
    #数据提取
    data = np.array(data_list[i],'f')
    sc = MinMaxScaler()
    data = np.reshape(data,(-1,1))
    data = sc.fit_transform(data)
    data = np.reshape(data,(-1))
    #切分
    X_train = data[:train_num]
    X_test = data[train_num:-1]
    y_train = data[1:train_num+1]
    y_test = data[train_num+1:]
    
    x = getWindow(X_train,window_size)
    x  = np.reshape(x,(x.shape[0],x.shape[-1]))
    y_train = y_train[window_size:]
    
    begin = time.time()
    clf = SVR(kernel='rbf',gamma=0.1,C=1e2)
    clf.fit(x,y_train)
    
    #进入在线训练环节
    predict_counter = 0
    y_pred = []
    for t in range(train_num+1,len(data)-1):
        predict_counter += 1
        #预测当前值
        #打包之前的内容去预测
        passX = data[t-window_size:t]
        passX = np.reshape(passX,(1,-1))
        prediction = clf.predict(passX)
        y_pred.append(prediction.item())
        #开始训练
        if predict_counter >= predict_window:
            training_data = getWindow(data[t-2*window_size:t],window_size)
            training_data = np.reshape(training_data,(training_data.shape[0],training_data.shape[-1]))
            label = data[t-window_size+1:t+1]
            predict_counter = 0
            clf.fit(training_data,label)
    
    y_pred = np.array(y_pred)
    end = time.time()
    print('time :{}s'.format(end-begin))
    y_true = data[len(data)-len(y_pred):]
    y_true = getBack(y_true,sc)
    y_pred = getBack(y_pred,sc)
    
    plt.figure()
    plt.plot(y_true, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()
    print(mean_squared_error(y_true,y_pred))
    