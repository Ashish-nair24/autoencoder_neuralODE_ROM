# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:49:19 2023

@author: ashis
"""

import numpy as np 
import matplotlib.pyplot as plt

visc = 9
arr = np.load('../Datasets/Initial_cond/dataSet_{}.npy'.format(visc))


# import scipy.io as sio
# test = sio.loadmat('test.mat')
# test = test['uu']

fig, ax = plt.subplots()
CS = ax.contourf(arr, colormap='coolwarm')
ax.set_title('Visc = {}'.format(visc))
ax.set_xlabel('x')
ax.set_ylabel('t')
cbar = fig.colorbar(CS)
plt.show()
plt.savefig('Data_{}.png'.format(visc))
#plt.close()
