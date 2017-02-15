# googd references:
# http://nbviewer.jupyter.org/github/JWarmenhoven/Machine-Learning/blob/master/notebooks/Programming%20Exercise%202%20-%20Logistic%20Regression.ipynb

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

#Load txt file
#refer http://minus9d.hatenablog.com/entry/2016/03/24/233236
data = np.loadtxt('ex2data1.txt', delimiter = ',')

#get x and y
X = data[:,[0,1]]
y = data[:,2]
#m = the number of data
#n = the number of feature
m, n = X.shape

def plot_data(X , y):
  mask = (y == 1)
  X_positive = X[mask]
  X_negative = X[np.invert(mask)]
  #this is not working correctly.
  # plt.plot(X_positive, "o", color = "green")
  #plot require x and y value respectively
  plt.plot(X_positive[:,0], X_positive[:,1], "o", color = "red", label = "Admitted")
  plt.plot(X_negative[:,0], X_negative[:,1],"o", color = "blue", label = "Not admitted")
  plt.xlabel('exam1 score')
  plt.ylabel('exam2 score')
  plt.legend(loc = 'upper right')


#plot data
plot_data(X,y)

X = np.c_[np.ones((m,1)), X]
initial_theta =np.zeros(1 + n)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def cost_function(theta,X,y):
  m = len(y)
  h = sigmoid(X.dot(theta))
  # print 'h:', h
  J = (- y * np.log(h) - (1-y) * np.log(1-h)).sum() * 1/m
  # grad = 1/m * X.T.dot(h-y)
  return J

def gradient(theta,X,y):
  m = len(y)
  h = sigmoid(X.dot(theta))
  # 1/m * X.T.dot(h-y) is going to be wrong. I'm not sure why. because 1/m is an integer?
  grad = X.T.dot(h-y) * 1/m
  #I'm not sure why flatten is needed
  return grad.flatten()
  #I'm not still quite sure how to use minimize method
res = optimize.minimize(cost_function, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})

def predict(theta, x, threshold=0.5):
  p = sigmoid(x.dot(theta)) >= threshold
  return p.astype('int')
#all training data
p = predict(res.x,X)
print 'Train accuracy {}%'.format(100*sum(p == y)/len(y))

#decision boudary
plt.scatter(45, 85, s=60, marker='v', label='(45, 85)')
x1_min, x1_max = X[:,1].min(), X[:,1].max()
x2_min, x2_max = X[:,2].min(), X[:,2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)),xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths = 1, colors = 'b')














