import os
os.chdir('E:\\code\\homework\\timeseries\\web-traffic-time-series-forecasting')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('train_1.csv').fillna(0)

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

data = np.array(data_list[0],'f')#需要前面的日期信息
#生成一个更为有规律的序列，同样为500个左右
import numpy as np
import math
import random
x = np.linspace(1,len(data),len(data))
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

#进行FFT分解，尝试寻找dominating frequency
from numpy import fft
data = np.array(y,'f')
yf = fft.fft(data)
f = fft.fftfreq(data.shape[0])
xf = np.arange(len(yf))
plt.plot(f,yf,color='red')
plt.show()

#Keras LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import time

split_factor = 0.8
split_num = int(len(data_list[0])*split_factor)
cols = train.columns[1:-1]

begin = time.time()
data = np.array(y,'f')
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

window_size = 40
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