%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

w = math.pi/10
b1 = 10*w
b2 = 2*b1
x1 = w*np.arange(0,10,0.1)
x2 = w*np.arange(10,20,0.1)+b1
x3 = w*np.arange(20,30,0.1)+b2
y1 = np.array(list(map(math.sin,x1)))
y2 = 2*np.array(list(map(math.sin,x2)))
y3 = 4*np.array(list(map(math.sin,x3)))
y = np.concatenate((y1,y2,y3))
x = np.arange(0,30,0.1)

plt.figure() 
plt.plot(x,y,"b--")
plt.title("Line plot")
plt.show()