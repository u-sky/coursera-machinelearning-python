import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# load matlab file in python
# reference https://docs.scipy.org/doc/scipy/reference/tutorial/io.html

data = sio.loadmat('ex3data1.mat')

# try to get first 10 rows
sel = data['X'][np.arange(10),:]

def displayData:
