# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:39:37 2023

@author: ashis
"""

# Evaluating latent dimension model

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

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


gamma = 1e1
dt = 0.001953125

# File Paths
modelPath = '../Models/Initial_cond/Lat Space/Gamma={}/'.format(gamma)
LatTrainDataPath = '../Datasets/Initial_cond/Latent Space/Gamma={}/dataComb_lat.npy'.format(gamma)
LatTestDataPath  = '../Datasets/Initial_cond/Latent Space/Gamma={}/'.format(gamma)
resultsPath   = '../Results/K-S Equations/Initial_cond/Lat Space/'
 

# Parameter space (Initial Conditions)
testParams = [7,8,9]


# Loading Model
model = torch.load(modelPath+'latModel_{}.pt'.format(gamma))
model.eval()

# Loading latent space training data 
LatTrainData = np.load(LatTrainDataPath)
latDim = LatTrainData.shape[1]
LatTrainData = torch.tensor(LatTrainData, dtype=torch.float32)

# Stacking training and target data
nTrainSamp = 6 # Number of training samples
sampSize   = 5000 # Size of each sample

for i in range(nTrainSamp):
    
    if i==0:
        trainData = LatTrainData[0:(sampSize-1), :]
        targetRHSData = (LatTrainData[1:sampSize, :] - LatTrainData[0:(sampSize-1), :])/dt
        targetData = (LatTrainData[1:sampSize, :])
    else:
        trainData = torch.concatenate((trainData, LatTrainData[i*sampSize:((i+1)*sampSize-1), :]))
        targetRHSData = torch.concatenate((targetRHSData, ((LatTrainData[i*sampSize+1:((i+1)*sampSize), :]-LatTrainData[i*sampSize:((i+1)*sampSize-1), :])/dt)))
        targetData = torch.concatenate((targetData, ((LatTrainData[i*sampSize+1:((i+1)*sampSize), :]))))

trainRecon = trainData + dt * model(trainData)
trainLoss = reconLoss(targetData, trainRecon).detach().numpy()

# Stacking target data 
testInputDataList = []
testRHSDataList   = []
testOutputDataList = []
testReconList = []
testLoss = []
for i in range(len(testParams)):
    
    latArray = np.load(LatTestDataPath + 'dataSet_{}_lat.npy'.format(testParams[i]))
    testInputDataList.append(torch.tensor(latArray[0:sampSize-1,:], dtype=torch.float32))
    testOutputDataList.append(torch.tensor(latArray[1:sampSize,:], dtype=torch.float32))
    testReconList.append(testInputDataList[i] + dt * model(testInputDataList[i]))
    testLoss.append(reconLoss(testOutputDataList[i], testReconList[i]).detach().numpy())
    
    
# Comparison plots
fig, ax = plt.subplots(2,len(testParams))
for i in range(2):
    for j in range(len(testParams)):
        
        if i==0:
            for k in range(latDim):
                ax[i,j].plot(testOutputDataList[j][:,k])
            
            if j==0:
                ax[i,j].get_xaxis().set_visible(False)
            else:
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)
                            
            ax[i,j].set_title('{}'.format(testParams[j]))
            
        else:
            for k in range(latDim):
                ax[i,j].plot(testReconList[j][:,k].detach())

            if j==0:
                continue
            else:
                ax[i,j].get_yaxis().set_visible(False)


            
plt.suptitle('Reconstruction $\gamma$={}'.format(gamma))
#ax.axis("off")
plt.savefig(resultsPath + 'LatComparisonRecon_{}.png'.format(gamma))
plt.close()



# Evolving latent trajectories
testEvolveList = []
dtTest = dt

for i in range(len(testParams)):
    
    latArray = testInputDataList[i]
    y0       = latArray[0,:]
    y0       = y0[None,:]
    y        = y0
    reconArray = y0
    
    for j in range(sampSize):
        
        y += dt*model(y)
        reconArray = torch.concatenate((reconArray,y),dim=0)
        
    testEvolveList.append(reconArray)
    del reconArray


# Comparison plots
fig, ax = plt.subplots(2,len(testParams))
for i in range(2):
    for j in range(len(testParams)):
        
        if i==0:
            for k in range(latDim):
                ax[i,j].plot(testOutputDataList[j][:,k])
            
            if j==0:
                ax[i,j].get_xaxis().set_visible(False)
            else:
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)
                            
            ax[i,j].set_title('{}'.format(testParams[j]))
            
        else:
            for k in range(latDim):
                ax[i,j].plot(testEvolveList[j][:,k].detach())

            if j==0:
                continue
            else:
                ax[i,j].get_yaxis().set_visible(False)


            
plt.suptitle('Evolution $\gamma$={}'.format(gamma))
#ax.axis("off")
plt.savefig(resultsPath + 'LatEvolutionRecon_{}.png'.format(gamma))
plt.close()
    
        
    
    
    
    
    



