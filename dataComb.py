# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:57:47 2023

@author: ashis
"""

# Combining datasets for training 
import numpy as np


# List of training parameters
trainList = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]


# Reading and combining data
for visc in trainList:
    
    arr = np.load('../Datasets/Initial_cond/Full Space/dataSet_{}.npy'.format(visc))
    if visc == trainList[0]:
        arrComb = arr
        
    else:
        arrComb = np.concatenate((arrComb, arr), axis=0)
        

np.save('../Datasets/Initial_cond/dataComb.npy', arrComb)
            



