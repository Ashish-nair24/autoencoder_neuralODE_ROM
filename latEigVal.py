# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 23:12:23 2023

@author: ashis
"""

# Evaluating latent dimension model

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', size=22)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

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
dt = 1e-2#0.001953125
numLayers = 5

# File Paths
modelPath = '../Scripts/2D_atmospheric_nODE/'
LatTrainDataPath = '../Datasets/Atmospheric/latTraj_atmospheric.npy'#'../Datasets/Initial_cond/Latent Space/numLayers_Test/dataComb_lat_{}.npy'.format(numLayers)
#LatTestDataPath  = '../Datasets/Initial_cond/Latent Space/numLayers_Test/'
resultsPathBase   = './Scripts/2D_atmospheric_nODE/'
 

# Parameter space (Initial Conditions)
testParams = [23,24,25]



# Loading latent space training data 
LatTrainData = np.load(LatTrainDataPath)
latDim = LatTrainData.shape[1]
LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)
trainData = LatTrainData

# Stacking training and target data
nTrainSamp = 6 # Number of training samples
sampSize   = 5000 # Size of each sample

# for i in range(nTrainSamp):
    
#     if i==0:
#         trainData = LatTrainData[1000:(sampSize), :]
#     else:
#         trainData = torch.concatenate((trainData, LatTrainData[(i*sampSize+1000):((i+1)*sampSize), :]))


# # Stacking target data 
# testInputDataList = []
# for i in range(len(testParams)):
    
#     latArray = np.load(LatTestDataPath + 'dataSet_{}_lat_{}.npy'.format(testParams[i], numLayers))
#     testInputDataList.append(torch.tensor(latArray[1000:sampSize,:], dtype=torch.float32))
    

varLengthList = [4,49,499]
tLimList_varLen = []

# Initialising environment variables
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('gloo')

for varLength in varLengthList:



    ### Loading Model
    print(varLength)
    # varLength = 499
    # cutoffSamp = 1000
    #nODE    = torch.load(nODEPathBase+'couplednODE_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
    nODE    = torch.load(modelPath+'latModelnODE_varLen_{}.pt'.format(varLength), map_location=torch.device('cpu'))
    nODE.eval()

    nODE_dict = nODE.state_dict()
    nODE_dict_copy = copy.deepcopy(nODE_dict)
    del nODE
    
    model = FullyConnectedNet(200, 200, 1000, 5)
    for k in nODE_dict.keys():
        k_old = k
        if len(k.split('module.')) == 2:
            k_new = k.split('module.')[1]
        else:
            k_new = k.split('module.')[0]
        nODE_dict_copy[k_new] = nODE_dict_copy.pop(k_old)

    model.load_state_dict(nODE_dict_copy)


    # Eigen value plots 
    # Within training set
    eigsList = []
    tLimList = []
    for i in range(1000,1100):
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
        
        
    tLimList_varLen.append(tLimList)
        

fullSysTS = []
t = np.linspace(1700,2000,101)[:100] * dt
# Loading full-system time scales
with open("tLimList.txt", 'r') as f:
    
    for line in f:
        currLine = line.rstrip('\n')
        fullSysTS.append(float(currLine))
    

# Plotting nODE time-scales 
for i in range(len(varLengthList)):
    plt.semilogy(t, tLimList_varLen[i], label=str(varLengthList[i]+1))
    
# Plotting full-system time-scale
#plt.semilogy(t, fullSysTS[:100], label='Full System')
#plt.title('Limiting time-scale')
plt.ylabel('$t_{lim}$')
plt.xlabel('t')
plt.legend(title='$n_t$')
plt.tight_layout()
#plt.show()
plt.savefig('../Scripts/2D_atmospheric_nODE/' + 'TimeScaleComp.pdf')
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