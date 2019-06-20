import os
os.chdir('E:\\code\\homework\\timeseries\\web-traffic-time-series-forecasting')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train_1.csv').fillna(0)
train.head()

#统计全部网站，找到最大的10个
#找到访问量最大的十个网站，并保存在data_list中
sum_set = pd.DataFrame(train[['Page']])
sum_set['total'] = train.sum(axis=1)
sum_set = sum_set.sort_values('total',ascending=False)
top_pages = sum_set.index[0:10]
data_list = []
for _,index in enumerate(top_pages):
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[index,cols]
    data_list.append(data)
    days = [x for x in range(len(cols))]
    plt.plot(days,data)
    plt.show()

#使用pytorch构建LSTM#使用Pytorch构建LSTM
import torch
from torch import nn
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


#ARIMA model
from statsmodels.tsa.arima_model import ARIMA
import warnings
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

for i,index in enumerate(data_list):
    begin = time.time()
    data = np.array(data_list[i],'f')
    train_data = data[:split_num]
    #切分训练数据
    
    result = None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        try:
            arima = ARIMA(train_data,[2,1,4])
            result = arima.fit(disp=False)
        except:
            try:
                arima = ARIMA(train_data,[2,1,2])
                result = arima.fit(disp=False)
            except:
                print('\tARIMA failed')
    #print(result.params)
    pred = result.predict(2,len(data_list[i])-1,typ='levels')
    end = time.time()
    print('predict time:{}'.format(end-begin))
    x = [i for i in range(len(data_list[i]))]
    plt.plot(x,data,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.title('Page{}'.format(i))
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()

#Keras LSTM 单层无滑动窗口
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

for i,index in enumerate(data_list):
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
    #Keras LSTM
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(units = 8, activation = 'relu', input_shape = (None, 1))) #输入一个单元，隐层8个，单层
    regressor.add(Dense(units = 1))#全连接层
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(X_train, y_train, batch_size = 10, epochs = 10, verbose = 0)
    #进入测试环节
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    inputs = np.reshape(inputs,(-1,1,1))
    y_pred = regressor.predict(inputs)
    y_pred = ysc.inverse_transform(y_pred)
    end = time.time()
    #计时结束
    print('total time for {} is {} s'.format(i,end-begin))
    plt.figure()
    plt.plot(y_test, color = 'red', label = 'Real Web View')
    plt.plot(y_pred, color = 'blue', label = 'Predicted Web View')
    plt.title('Web View Forecasting{}'.format(i))
    plt.xlabel('Number of Days from Start')
    plt.ylabel('Web View')
    plt.legend()
    plt.show()


#Keras 3层LSTM + 滑动窗口。普通版本
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

for i,index in enumerate(data_list):
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
    
    window_size = 10
    x = []
    for t in range(len(X_train)-window_size):
        a = X_train[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),1,window_size))
    y_train = y_train[window_size:]
    #Keras LSTM
    begin = time.time()
    regressor = Sequential()
    regressor.add(LSTM(8, return_sequences=True, input_shape=(x.shape[1], window_size)))
    regressor.add(LSTM(8, return_sequences=True))
    regressor.add(LSTM(8))
    regressor.add(Dense(1))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = 8, epochs = 100, verbose = 0.2,shuffle=False)
    #进入测试环节
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    inputs = np.reshape(inputs,(-1,1,1))
    x = []
    for t in range(len(inputs)-window_size):
        a = inputs[t:t+window_size]
        x.append(a)
    x = np.array(x)
    x = np.reshape(x,(len(x),1,window_size))
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

#facebook prophet
from fbprophet import Prophet
split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

for i,index in enumerate(data_list):
    begin = time.time()
    df = pd.DataFrame(data_list[i])
    df.columns = ['y']
    df['ds'] = cols
    train_data = df[:split_num]
    predictor = Prophet()
    predictor.fit(train_data)
    #进行预测
    future = predictor.make_future_dataframe(periods = len(data_list[0])-split_num)
    forecast = predictor.predict(future)
    end = time.time()
    print('total time for {} is {}s'.format(i,end-begin))
    x1 = forecast['ds']
    y1 = forecast['yhat']
    y2 = forecast['yhat_lower']
    y3 = forecast['yhat_upper']
    ans = df['y']
    plt.plot(x1,ans,'m',label='origin')
    plt.plot(x1,y1,'r',label='predict')
    plt.plot(x1,y2)
    plt.plot(x1,y3)
    plt.show()

#机器学习方法，随机森林（整体思想）
from sklearn.ensemble import RandomForestRegressor
for i in range(1):
    index = data_list[i]
    
    begin = time.time()
    data = np.array(data_list[i],'f')
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
    #进行训练
    regressor = RandomForestRegressor(n_estimators=100,max_depth=5,n_jobs = -1)
    regressor.fit(X_train,y_train.ravel())
    #开始预测
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    y_pred = regressor.predict(inputs)
    y_pred = y_pred.reshape(-1,1)
    y_pred = ysc.inverse_transform(y_pred)
    
    end = time.time()
    print('total time for {} is {}s'.format(i,end-begin))
    plt.plot(y_pred,label='predict')
    plt.plot(y_test,label='true')
    plt.legend(loc='upper right')
    plt.show()
    

#ARIMA具体研究，使用顶层数据8
df = pd.DataFrame(data_list[8])
df.columns = ['y']
train_data = df[:split_num]
data = train_data
data.head()
# 时间序列是稳定的，意味着它的一些统计信息，比如平均值和方差等不怎么随时间变化
#使用Dickey-Fuller Test来检验stationary
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(train_data)
#Dickey-Fuller Test:负的越多，拒绝null假设的概率越大，序列稳定的可能性越高。小于某个值代表达到某个可能性
from statsmodels.tsa.stattools import adfuller
X = train_data.values.reshape(-1,1)
X = X.ravel()
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
#非常简单的时间序列拆分：线性数据+周期性数据+残差
from statsmodels.tsa.seasonal import seasonal_decompose
x = train_data.values.astype('int32').ravel()
decomposition = seasonal_decompose(x, model='multiplicative',freq = 7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.plot(trend)
plt.show()
#一般使用ACF和PACF来决定p和q的值
from statsmodels.tsa.stattools import acf, pacf
ts_log_diff = train_data.diff(1)
ts_log_diff.dropna(inplace=True)
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20)
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
#p与q的值取第一个过上置信边界的值


for i in range(10):
    df = pd.DataFrame(data_list[i])
    df.columns = ['y']
    train_data = df[:split_num]
    ts_log_diff = train_data.diff(1)
    ts_log_diff.dropna(inplace=True)
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20)
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    #p与q的值取第一个过上置信边界的值
    a = max(-1.96/np.sqrt(len(ts_log_diff),1.96/np.sqrt(len(ts_log_diff))
    p = 1
    q = 1
    for ele in lag_acf:
        if ele<a:
            break
        p+=1
    for ele in lag_pacf:
        if ele<a:
            break
        q+=1
    print(p,q)
    
#进行预测，并与diff1进行比较
from statsmodels.tsa.arima_model import ARIMA
x = train_data.values.astype('int32').ravel()
model = ARIMA(x, order=(4,1,4))  
results_AR = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log_diff.values.astype('int32').ravel())**2))
plt.show()


#尝试滑动窗口ARIMA
#制作滑动窗口，预测，然后放入？
window_size = 40
for i in range(1):
    index = data_list[i]

    begin = time.time()
    data = np.array(data_list[i],'f')
    
    #切分训练数据
    for t in range(len(data)-window_size):
        train_data = data[t:t+window_size]
        arima = ARIMA(train_data,[2,1,4])
        result = arima.fit(disp=False)
        pred = result.predict(2,len(data_list[i])-1,typ='levels')
    
    
    

    #print(result.params)
    
    end = time.time()
    print('predict time:{}'.format(end-begin))
    x = [i for i in range(len(data_list[i]))]
    plt.plot(x,data,label='Data')
    plt.plot(x[2:],pred,label='ARIMA Model')
    plt.title('Page{}'.format(i))
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.legend()
    plt.show()


#RandomForestRegressor 进行sliding window
from sklearn.ensemble import RandomForestRegressor
# RandomForestRegressor
for i,index in enumerate(data_list):
    begin = time.time()
    data = np.array(data_list[i],'f')
    sc = MinMaxScaler()
    data = np.reshape(data,(-1,1))
    data = sc.fit_transform(data)
    window_size = 20
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
    y_test = data[window_size+2:]
    end = time.time()
    print('total time for {} is {}s'.format(i,end-begin))
    plt.plot(y_pred,label='predict')
    plt.plot(y_test,label='true')
    plt.legend(loc='upper right')
    plt.show()
    

#生成一个更为有规律的序列，同样为500个左右
import numpy as np
import math
x = np.linspace(1,500,500)
para = np.random.rand(3,10)
A = para[0]
w = para[1]
b = para[2]
y = np.zeros(500)
for i,_ in enumerate(y):
    ans = 0
    for j in range(len(A)):
        ans += A[j] * math.sin(w[j]*x[i]+b[j])
    y[i] = ans + random.randint(int(-0.05*ans),int(0.05*ans))