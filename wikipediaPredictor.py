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


#Keras 3层LSTM + 滑动窗口
#效果上而言预测精度堪忧
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
    
    window_size = 48
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
    regressor.add(LSTM(32, return_sequences=True, input_shape=(x.shape[1], window_size)))
    regressor.add(LSTM(32, return_sequences=True))
    regressor.add(LSTM(32))
    regressor.add(Dense(1))
    regressor.compile(loss='mean_squared_error',optimizer='adam')
    regressor.fit(x, y_train, batch_size = 32, epochs = 100, verbose = 0.2,shuffle=False)
    #进入测试环节
    inputs = X_test
    inputs = np.reshape(inputs,(-1,1))
    inputs = xsc.transform(inputs)
    inputs = np.reshape(inputs,(-1,1,1))
    window_size = 48
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