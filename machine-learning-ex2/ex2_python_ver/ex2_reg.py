import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from scipy import optimize

data = np.loadtxt('ex2data2.txt', delimiter = ',')
X = data[:,0:2]
y = data[:,2]
# m, n = X.shape


def plot_data(X, y):
  mask = y == 1
  X_pos = X[mask]
  X_neg = X[np.invert(mask)]
  plt.scatter(X_pos[:,0], X_pos[:,1], s = 20, marker='+', color='blue', label='y=1')
  plt.scatter(X_neg[:,0], X_neg[:,1], s = 20, marker='o', color='yellow', label='y=0')
  plt.xlabel('Microchip Test 1')
  plt.ylabel('Microchip Test 2')
  plt.legend(loc = 'upper right')

plot_data(X,y)

poly = PolynomialFeatures(6)
XX = poly.fit_transform(X)
# XX.shape

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# def cost_function_reg(theta, X, y, lamda):
#   m = len(y)
#   h = sigmoid(X.dot(theta))
#   J = (- y * np.log(h) - (1-y) * np.log(1-h)).sum() * 1/m + (theta ** 2).sum() * lamda/ 2 * m
#   return J



# i just coppied and pasted this because my contour looks wrong. (res.x will be zeros...) but still not working
def costFunctionReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    print J
    # if np.isnan(J[0]):
    #     return(np.inf)
    # return(J[0])



# def gradient_reg(theta, X, y, lamda):
#   m = len(y)
#   h = sigmoid(X.dot(theta))

#   # first theta should NOT be regularized
#   theta[0] = 0
#   grad = X.T.dot(h-y) * 1/m + theta * (lamda / m)
#   return grad.flatten()


# I just coppied and pasted this because my contour looks wrong. (res.x will be zeros...) but still not working
def gradientReg(theta, reg, *args):
    m = y.size
    h = sigmoid(XX.dot(theta.reshape(-1,1)))
      
    grad = (1/m)*XX.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())



def predict(theta, x, threshold=0.5):
  p = sigmoid(x.dot(theta)) >= threshold
  return p.astype('int')
# X = np.c_[np.ones((m,1)), X]
initial_theta = np.zeros(XX.shape[1])

# cost_function_reg(initial_theta, XX, y, 1)

# res = optimize.minimize(cost_function_reg, initial_theta, args=(XX,y,1), method = None, jac=gradient_reg, options={'maxiter': 3000})

res = optimize.minimize(costFunctionReg, initial_theta, args=(1, XX, y), method=None, jac=gradientReg, options={'maxiter':3000})
X1_min, X1_max = X[:,0].min(), X[:,0].max()
X2_min, X2_max = X[:,1].min(), X[:,1].max()
xx1, xx2 = np.meshgrid(np.linspace(X1_min, X1_max), np.linspace(X2_min, X2_max))
h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
# xx1.shpae and xx2.shape are [50,50]
# h.shape is [2500]
h = h.reshape(xx1.shape)

plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors="g")
