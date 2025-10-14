# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:47:33 2023

@author: ashis
"""


# Time Scale vs NN-depth

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint
import os 
import torch.distributed as dist
import copy 

def reconLoss(recon, data):
    
    return (torch.norm(recon-data)) / latDim


# NN-architecture
class FullyConnectedNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(FullyConnectedNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_size, hidden_size))
            else:
                self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # ELU activation
        self.activation = nn.ELU()
    
    def forward(self, t, y):
        # Pass input through hidden layers
        for i in range(self.num_layers):
            y = self.hidden_layers[i](y)
            y = self.activation(y)
        
        # Pass through output layer
        y = self.output_layer(y)
        
        return y

gamma = 0.0
dt = 0.001953125

# File Paths
modelPath = '../Models/Initial_cond/numLayers_Test/'
resultsPathBase   = '../Results/K-S Equations/Initial_cond/numLayers_Test/'


nLayersList = [2,3,5,6,8]
tLimList_nLayers = []


# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')

for nLayers in nLayersList:

    LatTrainDataPath = '../Datasets/Initial_cond/Latent Space/numLayers_Test/dataComb_lat_{}.npy'.format(nLayers)


    # Loading latent space training data 
    LatTrainData = np.load(LatTrainDataPath)
    latDim = LatTrainData.shape[1]
    LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)
    
    # Stacking training and target data
    nTrainSamp = 6 # Number of training samples
    sampSize   = 5000 # Size of each sample
    
    for i in range(nTrainSamp):
        
        if i==0:
            trainData = LatTrainData[1000:(sampSize), :]
        else:
            trainData = torch.concatenate((trainData, LatTrainData[(i*sampSize+1000):((i+1)*sampSize), :]))
    
    
    ### Loading Model
    # varLength = 499
    # cutoffSamp = 1000
    model = torch.load(modelPath+'latModelnODE_{}_varLen_999_numLayers_{}.pt'.format(gamma,nLayers), map_location=torch.device('cpu'))
    model.eval()

    nODE_dict = model.state_dict()
    nODE_dict_copy = copy.deepcopy(nODE_dict)
    del model
    
    model = FullyConnectedNet(10, 10, 100, 5)
    for k in nODE_dict.keys():
        k_old = k
        k_new = k.split('module.')[1]
        nODE_dict_copy[k_new] = nODE_dict_copy.pop(k_old)

    model.load_state_dict(nODE_dict_copy)
    
    
    # Eigen value plots 
    # Within training set
    eigsList = []
    tLimList = []
    for i in range(1700,2000):
        currentSamp = trainData[i, :]
        currentSamp = currentSamp[None, :]
        
        # Jacobian
        JacCurr = torch.autograd.functional.jacobian(model, (torch.DoubleTensor(0), currentSamp))
        JacCurr = JacCurr[1]
        JacCurr = JacCurr.squeeze()
        
        # Eigen values
        eigs = torch.linalg.eigvals(JacCurr)
        
        eigs = eigs.numpy()
        
        maxEig = np.amax(eigs.real)
        
        eigsList.append(maxEig)
        
        tLim = 1/maxEig
        
        tLimList.append(tLim)
        
        
    tLimList_nLayers.append(tLimList)
        

fullSysTS = []
t = np.linspace(1700,2000,301)[:300] * dt
# Loading full-system time scales
with open("tLimList.txt", 'r') as f:
    
    for line in f:
        currLine = line.rstrip('\n')
        fullSysTS.append(float(currLine))
    

# Plotting nODE time-scales 
for i in range(len(nLayersList)):
    plt.semilogy(t, tLimList_nLayers[i], label=str(nLayersList[i]))
    
# Plotting full-system time-scale
plt.semilogy(t, fullSysTS[:300], label='Full System')
plt.title('Limiting time-scale')
plt.ylabel('$t_{lim}$')
plt.xlabel('t')
plt.legend(title='Num Layers')
plt.savefig(resultsPathBase + 'TimeScaleVsNumLayers.png')
plt.close()


    

# # Plotting eigen values
# plt.plot(eigsList)
# plt.ylabel('Max eig')
# plt.savefig(resultsPath + 'eigValTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()

    
# plt.semilogy(tLimList)
# plt.ylabel('Limiting time-scale')
# plt.savefig(resultsPath + 'tLimTraining_{}_varLen_{}.png'.format(gamma, varLength))
# plt.close()