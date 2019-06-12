import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
import math
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

class Predictor:
    def Keras_RNN_fit(self,x,y):
        '''
        使用keras的RNN组建完成预测
        输入的为numpy.array，要求为一维数组且长度相同
        '''
        epoch = 10


    def Keras_RNN_predict(self,x):
        '''
        输入的为numpy.array
        '''

'''
totalNum = unit/factor

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True, stateful=True))
model.add(LSTM(32, stateful=True))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,
          validation_data=(x_val, y_val))
'''
#begin there

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

#留档
x_origin = x
x_origin = np.reshape(x_origin,(len(x_origin),1,w))
y_origin = y

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

#转换维度，使得数据可以被输入模型中

X_train = np.reshape(X_train,(len(X_train),1,w))
X_test = np.reshape(X_test,(len(X_test),1,w))

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

fit = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.2, shuffle=False)

pred_rnn = model.predict(X_test,batch_size=32)

y_pred_rnn = pred_rnn * x_std + x_mean
fithistory = pd.DataFrame(fit.history)

plt.figure()
plt.plot(range(1,101), fithistory['val_loss'],c='red', label='Loss(development)')
plt.plot(range(1,101), fithistory['loss'],c='blue',label='Loss(train)')
plt.show()

y_pred = model.predict(x_origin,batch_size=32)
y_pred = y_pred.reshape(len(y_pred))
y_pred = y_pred*x_std+x_mean
y_origin = y_origin*x_std+x_mean
x = list(range(len(y_pred)))

plt.figure()
plt.plot(x,y_pred,c='red')
plt.plot(x,y_origin,c='blue')
plt.show()

